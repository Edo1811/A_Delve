// ============================================================
//  Delve Launcher  —  src/launcher.cpp
// ============================================================
//  CONFIGURE THESE before each release / distribution:
// ============================================================
#define CURRENT_VERSION     "0.1.0"
// Plain-text file at this URL must contain ONLY the latest version string, e.g. "0.1.1"
#define UPDATE_HOST         L"raw.githubusercontent.com"
#define VERSION_PATH        L"/YOUR_GITHUB_USER/A_Delve/main/version.txt"
// Download URL for the installer — %S is wide-string version (e.g. "0.1.1")
#define DL_HOST             L"github.com"
#define DL_PATH_FMT         L"/YOUR_GITHUB_USER/A_Delve/releases/download/v%S/DelveSetup-%S.exe"
#define GAME_EXE            "Delve.exe"
// ============================================================

// ── Include order is critical on MinGW ──────────────────────────────────────
// CloseWindow, ShowCursor, DrawText, DrawTextEx, LoadImage are all real C
// function declarations in winuser.h — you cannot #undef a declaration.
// The only reliable fix is to rename them at the preprocessor level BEFORE
// windows.h is parsed, so winuser.h declares _w32_CloseWindow etc. instead.
// Then we undef the renames so raylib.h can declare the real symbols cleanly.
#define CloseWindow   _w32_CloseWindow
#define ShowCursor    _w32_ShowCursor
#define DrawText      _w32_DrawText
#define DrawTextEx    _w32_DrawTextEx
#define LoadImage     _w32_LoadImage

#define WIN32_LEAN_AND_MEAN   // strips most bloat
#define NOGDI                 // removes Rectangle / other GDI collisions
#define NOMINMAX              // no min/max macros
#include <windows.h>
#include <winhttp.h>
#include <shellapi.h>

// Now restore the names — raylib.h will declare the real versions below
#undef CloseWindow
#undef ShowCursor
#undef DrawText
#undef DrawTextEx
#undef LoadImage

#include "raylib.h"
#include "raymath.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstring>

// ── Palette (matches the game) ───────────────────────────────────────────────
static const Color C_BG       = {10, 8, 18, 255};
static const Color C_PANEL    = {18, 15, 10, 230};
static const Color C_BORDER   = {75, 68, 52, 255};
static const Color C_GOLD     = {215,175, 55, 255};
static const Color C_GOLD_DIM = {140,115, 40, 200};
static const Color C_TEXT     = {200,185,140, 230};
static const Color C_TEXT_DIM = {130,120, 95, 180};
static const Color C_GREEN    = { 80,200, 80, 255};
static const Color C_RED      = {210, 70, 70, 255};
static const Color C_BLUE     = { 80,160,230, 255};

// ── News entries (update this array each release) ────────────────────────────
struct NewsEntry { const char* version; const char* date; const char* items[8]; int nItems; };
static const NewsEntry NEWS[] = {
    { "v0.1.0", "2026-03-17",
      { "Initial release — Delve is live!",
        "Procedural cave system with ore veins",
        "Mining, inventory, crafting foundation",
        "Torch placement and dynamic lighting",
        "God-ray post-processing shader",
        "Day/night cycle with sunset sky",
        "Async chunk streaming — smooth loading",
        "In-game settings: FOV, FPS cap, sensitivity" },
      8 },
};
static const int NUM_NEWS = (int)(sizeof(NEWS)/sizeof(NEWS[0]));

// ── Launcher state ─────────────────────────────────────────────────────────
enum UpdateStatus { US_CHECKING, US_UP_TO_DATE, US_AVAILABLE, US_DOWNLOADING, US_DONE, US_ERROR, US_OFFLINE };

static std::atomic<UpdateStatus> gStatus{US_CHECKING};
static std::atomic<float>        gDlProgress{0.0f};    // 0..1
static std::atomic<int64_t>      gDlBytes{0};
static std::atomic<int64_t>      gDlTotal{0};
static std::mutex                gMtx;
static std::string               gLatestVersion;        // set by check thread
static std::string               gDlPath;               // temp installer path
static std::string               gErrorMsg;

// ── WinHTTP: synchronous GET, returns body as string.  Empty = failure. ─────
static std::string HttpGet(LPCWSTR host, LPCWSTR path){
    std::string result;
    HINTERNET hSess = WinHttpOpen(L"DelveLauncher/1.0",
                                  WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                  WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if(!hSess) return result;

    HINTERNET hConn = WinHttpConnect(hSess, host, INTERNET_DEFAULT_HTTPS_PORT, 0);
    if(!hConn){ WinHttpCloseHandle(hSess); return result; }

    HINTERNET hReq = WinHttpOpenRequest(hConn, L"GET", path, NULL,
                                        WINHTTP_NO_REFERER,
                                        WINHTTP_DEFAULT_ACCEPT_TYPES,
                                        WINHTTP_FLAG_SECURE);
    if(hReq && WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                  WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
            && WinHttpReceiveResponse(hReq, NULL)){
        char buf[4096]; DWORD read=0;
        while(WinHttpReadData(hReq, buf, sizeof(buf)-1, &read) && read>0){
            buf[read]=0; result += buf;
        }
    }
    if(hReq)  WinHttpCloseHandle(hReq);
    WinHttpCloseHandle(hConn);
    WinHttpCloseHandle(hSess);
    return result;
}

// ── WinHTTP: download URL to file, updating gDlBytes / gDlTotal ─────────────
static bool HttpDownload(LPCWSTR host, LPCWSTR path, const std::string& outPath){
    HINTERNET hSess = WinHttpOpen(L"DelveLauncher/1.0",
                                  WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                  WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if(!hSess) return false;

    HINTERNET hConn = WinHttpConnect(hSess, host, INTERNET_DEFAULT_HTTPS_PORT, 0);
    HINTERNET hReq  = nullptr;
    FILE*     fp    = nullptr;
    bool      ok    = false;

    if(!hConn) goto cleanup;
    hReq = WinHttpOpenRequest(hConn, L"GET", path, NULL, WINHTTP_NO_REFERER,
                              WINHTTP_DEFAULT_ACCEPT_TYPES,
                              WINHTTP_FLAG_SECURE);
    if(!hReq) goto cleanup;
    if(!WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                           WINHTTP_NO_REQUEST_DATA, 0, 0, 0)) goto cleanup;
    if(!WinHttpReceiveResponse(hReq, NULL)) goto cleanup;

    // Read Content-Length
    {
        WCHAR lenBuf[64]={0}; DWORD lenSz=sizeof(lenBuf);
        if(WinHttpQueryHeaders(hReq, WINHTTP_QUERY_CONTENT_LENGTH,
                               WINHTTP_HEADER_NAME_BY_INDEX, lenBuf, &lenSz,
                               WINHTTP_NO_HEADER_INDEX)){
            gDlTotal = (int64_t)_wtoi64(lenBuf);
        }
    }

    fp = fopen(outPath.c_str(), "wb");
    if(!fp) goto cleanup;

    {
        char buf[65536]; DWORD read=0;
        while(WinHttpReadData(hReq, buf, sizeof(buf), &read) && read>0){
            fwrite(buf, 1, read, fp);
            gDlBytes += read;
            int64_t tot = gDlTotal.load();
            if(tot>0) gDlProgress = (float)gDlBytes.load()/(float)tot;
        }
        ok = (gDlBytes.load() > 0);
    }

cleanup:
    if(fp)    fclose(fp);
    if(hReq)  WinHttpCloseHandle(hReq);
    if(hConn) WinHttpCloseHandle(hConn);
    if(hSess) WinHttpCloseHandle(hSess);
    return ok;
}

// ── Trim whitespace from both ends ──────────────────────────────────────────
static std::string Trim(const std::string& s){
    int a=0, b=(int)s.size()-1;
    while(a<=b && (s[a]==' '||s[a]=='\n'||s[a]=='\r'||s[a]=='\t')) a++;
    while(b>=a && (s[b]==' '||s[b]=='\n'||s[b]=='\r'||s[b]=='\t')) b--;
    return (a<=b) ? s.substr(a, b-a+1) : "";
}

// ── Compare two version strings "X.Y.Z" — returns true if b > a ─────────────
static bool VersionNewer(const std::string& a, const std::string& b){
    int av[3]={0,0,0}, bv[3]={0,0,0};
    sscanf(a.c_str(),"%d.%d.%d",&av[0],&av[1],&av[2]);
    sscanf(b.c_str(),"%d.%d.%d",&bv[0],&bv[1],&bv[2]);
    for(int i=0;i<3;i++){
        if(bv[i]>av[i]) return true;
        if(bv[i]<av[i]) return false;
    }
    return false;
}

// ── Background thread: check for update ────────────────────────────────────
static void CheckUpdateThread(){
    std::string body = HttpGet(UPDATE_HOST, VERSION_PATH);
    if(body.empty()){ gStatus=US_OFFLINE; return; }
    std::string latest = Trim(body);
    { std::lock_guard<std::mutex> lk(gMtx); gLatestVersion=latest; }
    if(VersionNewer(CURRENT_VERSION, latest))
        gStatus = US_AVAILABLE;
    else
        gStatus = US_UP_TO_DATE;
}

// ── Background thread: download installer ──────────────────────────────────
static void DownloadThread(){
    std::string ver;
    { std::lock_guard<std::mutex> lk(gMtx); ver = gLatestVersion; }

    // Build wide path string
    wchar_t wPath[512];
    swprintf(wPath, 512, DL_PATH_FMT,
             std::wstring(ver.begin(),ver.end()).c_str(),
             std::wstring(ver.begin(),ver.end()).c_str());

    // Temp file
    char tmp[MAX_PATH]; GetTempPathA(MAX_PATH,tmp);
    std::string outPath = std::string(tmp) + "DelveSetup-" + ver + ".exe";
    { std::lock_guard<std::mutex> lk(gMtx); gDlPath = outPath; }

    gDlBytes=0; gDlTotal=0; gDlProgress=0.0f;
    bool ok = HttpDownload(DL_HOST, wPath, outPath);
    if(ok) gStatus = US_DONE;
    else { gStatus = US_ERROR; std::lock_guard<std::mutex> lk(gMtx); gErrorMsg="Download failed."; }
}

// ── Procedural stone background texture ────────────────────────────────────
static Texture2D GenStoneTex(){
    const int W=128, H=128;
    Image img = GenImageColor(W,H,C_BG);
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        int h = x*1619 + y*31337 + 7;
        h = (h^(h>>13))*1664525 + 1013904223;
        float n  = (float)((h>>8)&0xFF)/255.0f * 0.18f;
        float cr = (float)((((x*374761393+y*914729)^((x*374761393+y*914729)>>13))*1664525+1013904223>>8)&0xFF)/255.0f;
        float crack = (cr > 0.91f) ? 0.68f : 1.0f;
        unsigned char v = (unsigned char)Clamp((14 + n*24)*crack, 0.0f, 255.0f);
        unsigned char r = (unsigned char)Clamp(v*0.92f, 0.0f, 255.0f);
        unsigned char g = (unsigned char)Clamp(v*0.85f, 0.0f, 255.0f);
        unsigned char b = (unsigned char)Clamp(v*0.75f, 0.0f, 255.0f);
        ImageDrawPixel(&img, x, y, {r,g,b,255});
    }
    Texture2D t = LoadTextureFromImage(img);
    SetTextureFilter(t, TEXTURE_FILTER_POINT);
    SetTextureWrap(t, TEXTURE_WRAP_REPEAT);
    UnloadImage(img);
    return t;
}

// ── Draw a bordered panel ───────────────────────────────────────────────────
static void DrawPanel(int x, int y, int w, int h){
    DrawRectangle(x, y, w, h, C_PANEL);
    DrawRectangleLines(x, y, w, h, C_BORDER);
    DrawRectangleLines(x+1, y+1, w-2, h-2, {40,36,26,120});
    // top highlight
    DrawLine(x+2, y+1, x+w-2, y+1, {80,72,52,80});
}

// ── Button — returns true on left-click release ─────────────────────────────
static bool DrawButton(int x, int y, int w, int h, const char* label, Color accent, bool enabled=true){
    Vector2 mp = GetMousePosition();
    bool over = enabled && mp.x>=x && mp.x<x+w && mp.y>=y && mp.y<y+h;

    Color fill   = over ? Color{(unsigned char)Clamp(accent.r*0.28f,0,255),
                                (unsigned char)Clamp(accent.g*0.28f,0,255),
                                (unsigned char)Clamp(accent.b*0.28f,0,255),240}
                        : Color{18,15,10,220};
    Color border = over ? accent : C_BORDER;
    if(!enabled){ fill={20,18,14,160}; border={50,45,35,120}; }

    DrawRectangle(x,y,w,h,fill);
    DrawRectangleLines(x,y,w,h,border);
    if(over) DrawRectangle(x,y,w,h,{accent.r,accent.g,accent.b,18});
    // inner highlight
    DrawLine(x+2,y+1,x+w-2,y+1,{255,240,200,(unsigned char)(over?60:20)});

    Color tc = enabled ? (over ? accent : C_TEXT) : C_TEXT_DIM;
    int fs=18, tw=MeasureText(label,fs);
    DrawText(label, x+1+w/2-tw/2, y+1+h/2-fs/2, fs, {0,0,0,100});
    DrawText(label,   x+w/2-tw/2,   y+h/2-fs/2,   fs, tc);

    return over && IsMouseButtonReleased(MOUSE_LEFT_BUTTON);
}

// ── Progress bar ───────────────────────────────────────────────────────────
static void DrawProgressBar(int x, int y, int w, int h, float t){
    DrawRectangle(x,y,w,h,{28,24,16,200});
    DrawRectangleLines(x,y,w,h,C_BORDER);
    int filled = (int)(t*(w-2));
    if(filled>0){
        DrawRectangle(x+1,y+1,filled,h-2,C_GOLD);
        DrawRectangle(x+1,y+1,filled,2,  {255,240,180,80}); // shine
    }
}

// ── Spinner animation ──────────────────────────────────────────────────────
static const char* Spinner(){ 
    static const char* f[]={"⠋","⠙","⠸","⠴","⠦","⠇"};
    return f[((int)(GetTime()*6))%6];
}

// ── News / Changelog panel ─────────────────────────────────────────────────
static float gNewsScroll = 0.0f;

static void DrawNewsPanel(int x, int y, int w, int h){
    DrawPanel(x,y,w,h);

    // Header
    const char* hdr="PATCH NOTES"; int hfs=16;
    DrawText(hdr, x+w/2-MeasureText(hdr,hfs)/2+1, y+13, hfs, {0,0,0,140});
    DrawText(hdr, x+w/2-MeasureText(hdr,hfs)/2,   y+12, hfs, C_GOLD);
    DrawLine(x+12,y+34,x+w-12,y+34,C_BORDER);

    // Scroll region — clip via scissors
    BeginScissorMode(x+8, y+38, w-16, h-46);
    int cy = y+42 - (int)gNewsScroll;
    for(int i=0;i<NUM_NEWS;i++){
        const NewsEntry& e = NEWS[i];
        // Version badge
        DrawRectangle(x+8, cy, 60, 18, C_GOLD);
        DrawText(e.version, x+9, cy+3, 12, {0,0,0,200});
        DrawText(e.date,    x+76, cy+3, 12, C_TEXT_DIM);
        cy+=24;
        // Bullet items
        for(int j=0;j<e.nItems;j++){
            DrawText("•", x+14, cy, 14, C_GOLD_DIM);
            DrawText(e.items[j], x+26, cy, 13, C_TEXT);
            cy+=18;
        }
        cy+=12;
        // Separator
        DrawLine(x+12, cy, x+w-12, cy, C_BORDER);
        cy+=10;
    }
    EndScissorMode();

    // Scroll via mouse wheel when hovering
    Vector2 mp=GetMousePosition();
    if(mp.x>=x && mp.x<x+w && mp.y>=y && mp.y<y+h){
        gNewsScroll -= GetMouseWheelMove()*20.0f;
        gNewsScroll = Clamp(gNewsScroll, 0.0f, (float)std::max(0, (cy-(y+h-46))));
    }
    // Scroll shadow fade
    DrawRectangleGradientV(x+8, y+38, w-16, 18, {18,15,10,200}, {18,15,10,0});
    DrawRectangleGradientV(x+8, y+h-24, w-16, 18, {18,15,10,0}, {18,15,10,200});
}

// ── Right panel: title, status, play button ─────────────────────────────────
// Returns true when the Play button is clicked.
static bool DrawRightPanel(int x, int y, int w, int h, std::atomic<bool>& downloadRequested){
    bool playClicked = false;
    DrawPanel(x,y,w,h);

    // ── Logo / title ──────────────────────────────────────────────────────
    int ty = y+24;
    const char* logo="DELVE"; int lfs=40;
    DrawText(logo, x+2+w/2-MeasureText(logo,lfs)/2, ty+2, lfs, {0,0,0,160});
    DrawText(logo,   x+w/2-MeasureText(logo,lfs)/2, ty,   lfs, C_GOLD);
    ty+=52;

    const char* sub="Mine. Craft. Conquer."; int sfs=13;
    DrawText(sub, x+w/2-MeasureText(sub,sfs)/2, ty, sfs, C_TEXT_DIM);
    ty+=28;

    DrawLine(x+16, ty, x+w-16, ty, C_BORDER); ty+=12;

    // ── Version badge ─────────────────────────────────────────────────────
    {
        char buf[64]; snprintf(buf,64,"Installed  v%s", CURRENT_VERSION);
        DrawText(buf, x+w/2-MeasureText(buf,13)/2, ty, 13, C_TEXT_DIM);
        ty+=18;

        UpdateStatus st = gStatus.load();
        if(st==US_CHECKING){
            char sb[64]; snprintf(sb,64,"%s Checking for updates...", Spinner());
            DrawText(sb, x+w/2-MeasureText(sb,12)/2, ty, 12, C_TEXT_DIM);
        } else if(st==US_UP_TO_DATE){
            const char* msg="Up to date  ✓";
            DrawText(msg, x+w/2-MeasureText(msg,13)/2, ty, 13, C_GREEN);
        } else if(st==US_AVAILABLE){
            std::string lv; { std::lock_guard<std::mutex> lk(gMtx); lv=gLatestVersion; }
            char msg[64]; snprintf(msg,64,"Update available  v%s",lv.c_str());
            DrawText(msg, x+w/2-MeasureText(msg,13)/2, ty, 13, C_BLUE);
        } else if(st==US_DOWNLOADING){
            char msg[64];
            int64_t tot=gDlTotal.load(), got=gDlBytes.load();
            if(tot>0) snprintf(msg,64,"Downloading…  %.1f / %.1f MB",
                               got/1e6, tot/1e6);
            else      snprintf(msg,64,"Downloading…  %.1f MB", got/1e6);
            DrawText(msg, x+w/2-MeasureText(msg,12)/2, ty, 12, C_BLUE);
        } else if(st==US_DONE){
            const char* msg="Download complete!";
            DrawText(msg, x+w/2-MeasureText(msg,13)/2, ty, 13, C_GREEN);
        } else if(st==US_ERROR){
            const char* msg="Update check failed";
            DrawText(msg, x+w/2-MeasureText(msg,13)/2, ty, 13, C_RED);
        } else if(st==US_OFFLINE){
            const char* msg="Offline — playing locally";
            DrawText(msg, x+w/2-MeasureText(msg,12)/2, ty, 12, C_TEXT_DIM);
        }
        ty+=22;
    }

    // ── Progress bar (during download) ────────────────────────────────────
    UpdateStatus st = gStatus.load();
    if(st==US_DOWNLOADING){
        DrawProgressBar(x+14, ty, w-28, 12, gDlProgress.load());
        ty+=20;
    }

    ty+=8;
    DrawLine(x+16, ty, x+w-16, ty, C_BORDER); ty+=16;

    // ── Action buttons ─────────────────────────────────────────────────────
    int bw=w-32, bx=x+16;

    if(st==US_AVAILABLE){
        // Update button
        if(DrawButton(bx, ty, bw, 38, "Download Update", C_BLUE)){
            if(!downloadRequested.load()){
                downloadRequested=true;
                gStatus=US_DOWNLOADING;
                std::thread(DownloadThread).detach();
            }
        }
        ty+=46;
        // Play anyway (smaller)
        if(DrawButton(bx, ty, bw, 30, "Play Current Version", C_TEXT_DIM)){
            playClicked=true;
        }
        ty+=38;
    } else if(st==US_DONE){
        // Install button
        if(DrawButton(bx, ty, bw, 38, "Install Update", C_BLUE)){
            std::string dlp; { std::lock_guard<std::mutex> lk(gMtx); dlp=gDlPath; }
            if(!dlp.empty()) ShellExecuteA(nullptr,"runas",dlp.c_str(),nullptr,nullptr,SW_SHOWNORMAL);
            CloseWindow();
        }
        ty+=46;
    } else {
        // Normal PLAY button
        bool canPlay = (st!=US_DOWNLOADING);
        if(DrawButton(bx, ty, bw, 50, "PLAY", C_GOLD, canPlay)){
            playClicked=true;
        }
        ty+=58;
    }

    // Quit link
    {
        const char* ql="Quit"; int qfs=13;
        int qx=x+w/2-MeasureText(ql,qfs)/2;
        Vector2 mp=GetMousePosition();
        bool hov=(mp.x>=qx&&mp.x<qx+MeasureText(ql,qfs)&&mp.y>=ty&&mp.y<ty+qfs+4);
        DrawText(ql, qx, ty, qfs, hov?C_TEXT:C_TEXT_DIM);
        if(hov && IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) CloseWindow();
        ty+=20;
    }

    // Bottom version footer
    {
        char footer[64]; snprintf(footer,64,"Delve v%s  —  © 2026", CURRENT_VERSION);
        int ffs=11;
        DrawText(footer, x+w/2-MeasureText(footer,ffs)/2, y+h-18, ffs, C_TEXT_DIM);
    }

    return playClicked;
}

// ── Launch Delve.exe in the same directory as the launcher ──────────────────
static bool LaunchGame(){
    char exePath[MAX_PATH]={};
    GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    // Replace launcher exe name with game exe name
    char* lastSlash = strrchr(exePath, '\\');
    if(lastSlash) *(lastSlash+1)='\0';
    else exePath[0]='\0';
    std::string gamePath = std::string(exePath) + GAME_EXE;

    STARTUPINFOA si={}; si.cb=sizeof(si);
    PROCESS_INFORMATION pi={};
    return CreateProcessA(gamePath.c_str(), nullptr, nullptr, nullptr, FALSE,
                          0, nullptr, exePath, &si, &pi) != 0;
}

// ── Entry point ─────────────────────────────────────────────────────────────
int main(){
    const int W=940, H=520;
    SetConfigFlags(FLAG_WINDOW_UNDECORATED);
    InitWindow(W, H, "Delve Launcher");
    SetTargetFPS(60);
    // Centre window on monitor
    int mon=GetCurrentMonitor();
    int mx=(GetMonitorWidth(mon)-W)/2, my=(GetMonitorHeight(mon)-H)/2;
    SetWindowPosition(mx,my);
    // Custom draggable titlebar: track drag
    Vector2 dragStart={0,0}; bool dragging=false;

    Texture2D stoneTex = GenStoneTex();

    // Kick off update check immediately in background
    std::thread(CheckUpdateThread).detach();
    std::atomic<bool> downloadRequested{false};
    bool shouldPlay=false;

    while(!WindowShouldClose()){
        // ── Dragging (undecorated window) ────────────────────────────────
        Vector2 mp=GetMousePosition();
        if(IsMouseButtonPressed(MOUSE_LEFT_BUTTON) && mp.y<34){
            dragging=true;
            dragStart=mp;
        }
        if(dragging){
            if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
                Vector2 winPos = GetWindowPosition();
                SetWindowPosition((int)(winPos.x + mp.x - dragStart.x),
                                  (int)(winPos.y + mp.y - dragStart.y));
                dragStart = mp;
            } else dragging=false;
        }

        if(shouldPlay){
            if(LaunchGame()) break;
            // If launch fails, show error but stay open
            shouldPlay=false;
        }

        BeginDrawing();
        ClearBackground(C_BG);

        // ── Stone tile background ─────────────────────────────────────────
        {
            float tw=(float)stoneTex.width, th=(float)stoneTex.height;
            for(int ty2=0;ty2<H;ty2+=(int)th)
            for(int tx2=0;tx2<W;tx2+=(int)tw)
                DrawTextureEx(stoneTex,{(float)tx2,(float)ty2},0,1.0f,{255,255,255,255});
        }
        // Dark vignette overlay
        DrawRectangleGradientEx({0,0,(float)W,(float)H},
                                {0,0,0,160},{0,0,0,80},{0,0,0,80},{0,0,0,160});

        // ── Custom title bar ──────────────────────────────────────────────
        DrawRectangle(0,0,W,32,{8,6,4,240});
        DrawLine(0,32,W,32,C_BORDER);
        const char* wtitle="Delve Launcher";
        DrawText(wtitle, W/2-MeasureText(wtitle,14)/2, 8, 14, C_TEXT_DIM);
        // Close button
        bool xhov=(mp.x>=W-32&&mp.x<W&&mp.y>=0&&mp.y<32);
        DrawText("✕", W-20, 9, 14, xhov?C_RED:C_TEXT_DIM);
        if(xhov && IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) break;

        // ── Panels ────────────────────────────────────────────────────────
        const int PAD=10, TOP=42;
        const int newsW=580, rightW=W-newsW-PAD*3;
        DrawNewsPanel(PAD, TOP, newsW, H-TOP-PAD);
        if(DrawRightPanel(PAD+newsW+PAD, TOP, rightW, H-TOP-PAD, downloadRequested))
            shouldPlay=true;

        EndDrawing();
    }

    UnloadTexture(stoneTex);
    CloseWindow();
    return 0;
}   