// ============================================================
//  Delve Launcher  —  src/launcher.cpp
// ============================================================
//  CONFIGURE THESE before each release / distribution:
// ============================================================
#define CURRENT_VERSION     "0.1.1"
#define UPDATE_HOST         L"raw.githubusercontent.com"
#define VERSION_PATH        L"/Edo1811/A_Delve/main/version.txt"
#define DL_HOST             L"github.com"
#define DL_PATH_FMT         L"/Edo1811/A_Delve/releases/download/v%S/DelveSetup-%S.exe"
#define GAME_EXE            "Delve.exe"
// ============================================================

#define CloseWindow   _w32_CloseWindow
#define ShowCursor    _w32_ShowCursor
#define DrawText      _w32_DrawText
#define DrawTextEx    _w32_DrawTextEx
#define LoadImage     _w32_LoadImage
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#include <shellapi.h>
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

// ── Window size ──────────────────────────────────────────────────────────────
static const int W = 980;
static const int H = 560;

// ── Palette ───────────────────────────────────────────────────────────────────
static const Color C_BG       = { 6,  5, 12, 255};
static const Color C_PANEL    = {14, 12,  9, 210};
static const Color C_BORDER   = {65, 58, 42, 200};
static const Color C_GOLD     = {220,178, 50, 255};
static const Color C_GOLD_DIM = {140,112, 35, 180};
static const Color C_GOLD_BRT = {255,220,100, 255};
static const Color C_TEXT     = {210,196,156, 230};
static const Color C_TEXT_DIM = {130,120, 90, 170};
static const Color C_GREEN    = { 70,200, 90, 255};
static const Color C_RED      = {210, 65, 65, 255};
static const Color C_BLUE     = { 70,155,235, 255};

// ── DPI / resolution scale ───────────────────────────────────────────────────
// Reference design resolution is 1920×1080. Everything is multiplied by gScale
// so the launcher looks identical at any resolution (1080p, 1440p, 4K, etc.).
static float gScale = 1.0f;
static int   S(float v){ return (int)(v * gScale + 0.5f); } // scaled int pixel value
static float Sf(float v){ return v * gScale; }               // scaled float

// ── Fonts (loaded at startup) ────────────────────────────────────────────────
static Font gFontTitle = {};   // large display font  (Segoe UI Black / fallback)
static Font gFontBody  = {};   // body / UI font      (Segoe UI / fallback)
static bool gFontsLoaded = false;

static void LoadFonts(){
    // Try Windows system fonts; gracefully fall back to raylib default if absent.
    // We load at several sizes to avoid blurry scaling.
    const char* titleFonts[] = {
        "C:/Windows/Fonts/seguibl.ttf",   // Segoe UI Black
        "C:/Windows/Fonts/segoeuib.ttf",  // Segoe UI Bold
        "C:/Windows/Fonts/arialbd.ttf",   // Arial Bold
        nullptr
    };
    const char* bodyFonts[] = {
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        nullptr
    };

    for(int i=0; titleFonts[i]; i++){
        if(FileExists(titleFonts[i])){
            gFontTitle = LoadFontEx(titleFonts[i], S(96), nullptr, 0);
            break;
        }
    }
    for(int i=0; bodyFonts[i]; i++){
        if(FileExists(bodyFonts[i])){
            gFontBody = LoadFontEx(bodyFonts[i], S(36), nullptr, 0);
            break;
        }
    }
    // Fall back to raylib default if nothing loaded
    if(gFontTitle.glyphCount == 0) gFontTitle = GetFontDefault();
    if(gFontBody.glyphCount  == 0) gFontBody  = GetFontDefault();
    SetTextureFilter(gFontTitle.texture, TEXTURE_FILTER_BILINEAR);
    SetTextureFilter(gFontBody.texture,  TEXTURE_FILTER_BILINEAR);
    gFontsLoaded = true;
}

// ── Font drawing helpers ─────────────────────────────────────────────────────
static void DrawBodyText(const char* text, int x, int y, float size, Color col){
    DrawTextPro(gFontBody, text, {(float)x,(float)y}, {0,0}, 0, size, size*0.04f, col);
}
static void DrawTitleText(const char* text, int x, int y, float size, Color col){
    DrawTextPro(gFontTitle, text, {(float)x,(float)y}, {0,0}, 0, size, size*0.03f, col);
}
static float MeasureBody(const char* text, float size){
    return MeasureTextEx(gFontBody, text, size, size*0.04f).x;
}
static float MeasureTitle(const char* text, float size){
    return MeasureTextEx(gFontTitle, text, size, size*0.03f).x;
}

// ── Centred helpers ──────────────────────────────────────────────────────────
static void DrawBodyCentre(const char* t, int cx, int y, float sz, Color c){
    DrawBodyText(t, cx-(int)(MeasureBody(t,sz)*0.5f), y, sz, c);
}
static void DrawTitleCentre(const char* t, int cx, int y, float sz, Color c){
    DrawTitleText(t, cx-(int)(MeasureTitle(t,sz)*0.5f), y, sz, c);
}

// ── News entries ──────────────────────────────────────────────────────────────
struct NewsEntry { const char* version; const char* date; const char* items[8]; int nItems; };
static const NewsEntry NEWS[] = {
    { "v0.1.0", "2026-03-17",
      { "Initial release — Delve is live!",
        "Procedural cave system with ore veins",
        "Mining, inventory and crafting foundation",
        "Torch placement and dynamic torch lighting",
        "God-ray post-processing shader",
        "Day/night cycle with atmospheric sky",
        "Async chunk streaming — smooth loading",
        "In-game settings: FOV, FPS cap, sensitivity" },
      8 },
};
static const int NUM_NEWS = (int)(sizeof(NEWS)/sizeof(NEWS[0]));

// ── Launcher state ────────────────────────────────────────────────────────────
enum UpdateStatus { US_CHECKING, US_UP_TO_DATE, US_AVAILABLE,
                    US_DOWNLOADING, US_DONE, US_ERROR, US_OFFLINE };
static std::atomic<UpdateStatus> gStatus{US_CHECKING};
static std::atomic<float>        gDlProgress{0.0f};
static std::atomic<int64_t>      gDlBytes{0};
static std::atomic<int64_t>      gDlTotal{0};
static std::mutex                gMtx;
static std::string               gLatestVersion;
static std::string               gDlPath;
static std::string               gErrorMsg;

// ── WinHTTP helpers ──────────────────────────────────────────────────────────
static std::string HttpGet(LPCWSTR host, LPCWSTR path){
    std::string result;
    HINTERNET hS=WinHttpOpen(L"DelveLauncher/1.0",WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                             WINHTTP_NO_PROXY_NAME,WINHTTP_NO_PROXY_BYPASS,0);
    if(!hS) return result;
    HINTERNET hC=WinHttpConnect(hS,host,INTERNET_DEFAULT_HTTPS_PORT,0);
    if(!hC){ WinHttpCloseHandle(hS); return result; }
    HINTERNET hR=WinHttpOpenRequest(hC,L"GET",path,NULL,WINHTTP_NO_REFERER,
                                    WINHTTP_DEFAULT_ACCEPT_TYPES,WINHTTP_FLAG_SECURE);
    if(hR&&WinHttpSendRequest(hR,WINHTTP_NO_ADDITIONAL_HEADERS,0,
                              WINHTTP_NO_REQUEST_DATA,0,0,0)
         &&WinHttpReceiveResponse(hR,NULL)){
        char buf[4096]; DWORD rd=0;
        while(WinHttpReadData(hR,buf,sizeof(buf)-1,&rd)&&rd>0){ buf[rd]=0; result+=buf; }
    }
    if(hR) WinHttpCloseHandle(hR);
    WinHttpCloseHandle(hC); WinHttpCloseHandle(hS);
    return result;
}
static bool HttpDownload(LPCWSTR host, LPCWSTR path, const std::string& out){
    HINTERNET hS=WinHttpOpen(L"DelveLauncher/1.0",WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                             WINHTTP_NO_PROXY_NAME,WINHTTP_NO_PROXY_BYPASS,0);
    if(!hS) return false;
    HINTERNET hC=WinHttpConnect(hS,host,INTERNET_DEFAULT_HTTPS_PORT,0);
    HINTERNET hR=nullptr; FILE* fp=nullptr; bool ok=false;
    if(!hC) goto cl;
    hR=WinHttpOpenRequest(hC,L"GET",path,NULL,WINHTTP_NO_REFERER,
                          WINHTTP_DEFAULT_ACCEPT_TYPES,WINHTTP_FLAG_SECURE);
    if(!hR||!WinHttpSendRequest(hR,WINHTTP_NO_ADDITIONAL_HEADERS,0,
                                WINHTTP_NO_REQUEST_DATA,0,0,0)
          ||!WinHttpReceiveResponse(hR,NULL)) goto cl;
    { WCHAR lb[64]={0}; DWORD ls=sizeof(lb);
      if(WinHttpQueryHeaders(hR,WINHTTP_QUERY_CONTENT_LENGTH,
                             WINHTTP_HEADER_NAME_BY_INDEX,lb,&ls,
                             WINHTTP_NO_HEADER_INDEX)) gDlTotal=(int64_t)_wtoi64(lb); }
    fp=fopen(out.c_str(),"wb"); if(!fp) goto cl;
    { char buf[65536]; DWORD rd=0;
      while(WinHttpReadData(hR,buf,sizeof(buf),&rd)&&rd>0){
          fwrite(buf,1,rd,fp); gDlBytes+=rd;
          int64_t t=gDlTotal.load();
          if(t>0) gDlProgress=(float)gDlBytes.load()/(float)t;
      } ok=(gDlBytes.load()>0); }
cl: if(fp) fclose(fp);
    if(hR) WinHttpCloseHandle(hR);
    if(hC) WinHttpCloseHandle(hC);
    WinHttpCloseHandle(hS); return ok;
}
static std::string Trim(const std::string& s){
    int a=0,b=(int)s.size()-1;
    while(a<=b&&(s[a]==' '||s[a]=='\n'||s[a]=='\r'||s[a]=='\t')) a++;
    while(b>=a&&(s[b]==' '||s[b]=='\n'||s[b]=='\r'||s[b]=='\t')) b--;
    return (a<=b)?s.substr(a,b-a+1):"";
}
static bool VersionNewer(const std::string& a, const std::string& b){
    int av[3]={},bv[3]={};
    sscanf(a.c_str(),"%d.%d.%d",&av[0],&av[1],&av[2]);
    sscanf(b.c_str(),"%d.%d.%d",&bv[0],&bv[1],&bv[2]);
    for(int i=0;i<3;i++){ if(bv[i]>av[i]) return true; if(bv[i]<av[i]) return false; }
    return false;
}
static void CheckUpdateThread(){
    std::string body=HttpGet(UPDATE_HOST,VERSION_PATH);
    if(body.empty()){ gStatus=US_OFFLINE; return; }
    std::string latest=Trim(body);
    { std::lock_guard<std::mutex> lk(gMtx); gLatestVersion=latest; }
    gStatus=VersionNewer(CURRENT_VERSION,latest)?US_AVAILABLE:US_UP_TO_DATE;
}
static void DownloadThread(){
    std::string ver; { std::lock_guard<std::mutex> lk(gMtx); ver=gLatestVersion; }
    wchar_t wPath[512];
    swprintf(wPath,512,DL_PATH_FMT,
             std::wstring(ver.begin(),ver.end()).c_str(),
             std::wstring(ver.begin(),ver.end()).c_str());
    char tmp[MAX_PATH]; GetTempPathA(MAX_PATH,tmp);
    std::string outPath=std::string(tmp)+"DelveSetup-"+ver+".exe";
    { std::lock_guard<std::mutex> lk(gMtx); gDlPath=outPath; }
    gDlBytes=0; gDlTotal=0; gDlProgress=0.0f;
    if(HttpDownload(DL_HOST,wPath,outPath)) gStatus=US_DONE;
    else { gStatus=US_ERROR; std::lock_guard<std::mutex> lk(gMtx); gErrorMsg="Download failed."; }
}

// ── Procedural background: dark cave panorama ────────────────────────────────
// Renders directly — no texture needed.
static void DrawBackground(float t, int SW, int SH){
    DrawRectangleGradientV(0,0,SW,SH, {4,3,8,255}, {8,6,18,255});

    for(int i=0;i<3;i++){
        float yf = SH*0.3f + sinf(t*0.2f+i*1.4f)*20.0f + i*60.0f;
        DrawRectangleGradientV(0,(int)yf,SW,40,
            {20,16,35,0},{20,16,35,(unsigned char)(8+i*4)});
        DrawRectangleGradientV(0,(int)(yf+40),SW,30,
            {20,16,35,(unsigned char)(8+i*4)},{20,16,35,0});
    }

    auto rng=[](int x,int s)->float{
        int h=x*1619+s*31337; h=(h^(h>>13))*1664525+1013904223;
        return (float)((h>>8)&0xFF)/255.0f;
    };
    for(int x=0;x<SW;x+=2){
        float base=rng(x/2,1)*0.15f + rng(x/2+100,2)*0.08f;
        DrawRectangle(x, 0, 2, (int)(base*110), {6,5,12,200});
    }
    for(int x=0;x<SW;x+=2){
        float base = 0.72f + rng(x/2,3)*0.12f + rng(x/2+50,4)*0.08f;
        int top=(int)(base*SH);
        DrawRectangle(x, top, 2, SH-top, {8,6,14,255});
        DrawRectangle(x, top, 2, 1, {35,28,50,120});
    }

    struct TorchGlow { float fx,fy,phase; Color col; };
    TorchGlow glows[]={
        {0.12f,0.72f,0.0f,{200,100,30,255}},
        {0.35f,0.76f,1.2f,{200,100,30,255}},
        {0.65f,0.74f,2.5f,{180,90,25,255}},
        {0.88f,0.70f,0.8f,{200,100,30,255}},
    };
    for(auto& g : glows){
        float flicker=0.7f+0.3f*sinf(t*4.3f+g.phase)*sinf(t*7.1f+g.phase*1.3f);
        int gx=(int)(g.fx*SW), gy=(int)(g.fy*SH);
        for(int r=80;r>0;r-=4){
            unsigned char a=(unsigned char)(Clamp(flicker*(1.0f-(float)r/80.0f)*28.0f,0,255));
            DrawCircle(gx,gy,r,{g.col.r,g.col.g,g.col.b,a});
        }
    }

    struct Particle { float x,y,life; };
    static std::vector<Particle> particles;
    static bool pInit=false;
    if(!pInit){ pInit=true;
        for(int i=0;i<60;i++){
            auto r=[&](int s)->float{ return rng(i*13+s,s+7); };
            particles.push_back({r(1)*(float)SW, r(2)*(float)SH, r(3)});
        }
    }
    for(auto& p : particles){
        p.y -= 0.08f; p.x += sinf(p.life*6.28f)*0.12f; p.life+=0.0004f;
        if(p.y<0||p.life>1.0f){ p.y=(float)SH; p.x=rng((int)(p.y+t*100),7)*(float)SW; p.life=0; }
        unsigned char a=(unsigned char)(Clamp(sinf(p.life*3.14f)*35.0f,0,255));
        DrawCircle((int)p.x,(int)p.y, 1, {200,185,150,a});
    }
}

// ── Panel with depth ─────────────────────────────────────────────────────────
static void DrawPanel(int x, int y, int w, int h, bool bright=false){
    // Drop shadow
    DrawRectangle(x+4, y+4, w, h, {0,0,0,80});
    // Fill
    unsigned char fa = bright ? 230 : 200;
    DrawRectangle(x, y, w, h, {14,12,9,fa});
    // Top gradient shine
    DrawRectangleGradientV(x,y,w,h/3, {255,240,180,8}, {255,240,180,0});
    // Outer border
    DrawRectangleLines(x,y,w,h, C_BORDER);
    // Inner border (subtle)
    DrawRectangleLines(x+1,y+1,w-2,h-2, {50,44,30,80});
    // Top highlight line
    DrawLine(x+2,y+1,x+w-3,y+1,{255,240,190,30});
}

// ── Gold separator line with glow ────────────────────────────────────────────
static void DrawGoldSep(int x, int y, int w){
    DrawLine(x,y+1,x+w,y+1,{0,0,0,60});
    DrawLine(x,y,  x+w,y,  {C_GOLD.r,C_GOLD.g,C_GOLD.b,60});
    // Central glow dot
    int cx=x+w/2;
    for(int r=8;r>0;r--)
        DrawRectangle(cx-r,y-r/2,r*2,r/2+1,
                      {C_GOLD.r,C_GOLD.g,C_GOLD.b,(unsigned char)(5*(8-r))});
    DrawCircle(cx,y,2,C_GOLD);
}

// ── Fancy play button ─────────────────────────────────────────────────────────
// Returns true on click.
static bool DrawPlayButton(int x, int y, int w, int h, bool enabled, float t){
    Vector2 mp=GetMousePosition();
    bool over=enabled&&mp.x>=x&&mp.x<x+w&&mp.y>=y&&mp.y<y+h;
    bool press=over&&IsMouseButtonDown(MOUSE_LEFT_BUTTON);

    // Outer glow when hovered
    if(over){
        for(int r=16;r>0;r-=2){
            unsigned char a=(unsigned char)(12*(1.0f-(float)r/16.0f));
            DrawRectangle(x-r,y-r,w+r*2,h+r*2,{C_GOLD.r,C_GOLD.g,C_GOLD.b,a});
        }
    }

    // Background
    float br = press?0.85f:(over?1.0f:0.7f);
    DrawRectangle(x,y,w,h,{(unsigned char)(40*br),(unsigned char)(32*br),(unsigned char)(10*br),240});

    // Animated shimmer sweep across button when hovered
    if(over){
        float shimX = fmodf(t*120.0f, (float)(w+80)) - 40.0f;
        for(int i=-20;i<20;i++){
            int sx=x+(int)shimX+i;
            if(sx<x||sx>=x+w) continue;
            unsigned char sa=(unsigned char)(Clamp((1.0f-fabsf((float)i/20.0f))*40.0f,0,255));
            DrawLine(sx,y,sx,y+h,{255,240,180,sa});
        }
    }

    // Gold border (thicker)
    Color bc = over ? C_GOLD_BRT : C_GOLD;
    DrawRectangleLines(x,y,w,h,bc);
    DrawRectangleLines(x+1,y+1,w-2,h-2,{bc.r,bc.g,bc.b,80});

    // Top shine
    DrawRectangleGradientV(x+2,y+2,w-4,h/2, {255,240,190,(unsigned char)(over?50:25)}, {255,240,190,0});

    // Label
    float fsz = Sf(26.0f);
    float tw = MeasureTitle("PLAY",fsz);
    int lx=(int)(x+w/2-tw/2), ly=y+h/2-(int)(fsz*0.5f);
    // Shadow
    DrawTitleText("PLAY", lx+1, ly+2, fsz, {0,0,0,120});
    // Glow
    if(over){
        DrawTitleText("PLAY", lx, ly, fsz, {255,240,180,60});
    }
    DrawTitleText("PLAY", lx, ly, fsz, over?C_GOLD_BRT:C_GOLD);

    if(!enabled){
        DrawRectangle(x,y,w,h,{0,0,0,100});
    }
    return over&&IsMouseButtonReleased(MOUSE_LEFT_BUTTON);
}

// ── Generic button ───────────────────────────────────────────────────────────
static bool DrawButton(int x, int y, int w, int h, const char* label,
                       Color accent, bool enabled=true, float sz=16.0f){
    Vector2 mp=GetMousePosition();
    bool over=enabled&&mp.x>=x&&mp.x<x+w&&mp.y>=y&&mp.y<y+h;

    if(over){ // outer glow
        for(int r=8;r>0;r-=2)
            DrawRectangle(x-r,y-r,w+r*2,h+r*2,
                          {accent.r,accent.g,accent.b,(unsigned char)(6*(1.0f-(float)r/8.0f)*255)});
    }

    unsigned char fa=enabled?(over?220:180):120;
    DrawRectangle(x,y,w,h,{18,15,10,fa});
    Color bc=enabled?(over?accent:C_BORDER):Color{40,36,26,120};
    DrawRectangleLines(x,y,w,h,bc);
    if(over) DrawRectangleGradientV(x+1,y+1,w-2,h/2,{accent.r,accent.g,accent.b,20},{0,0,0,0});
    DrawLine(x+2,y+1,x+w-2,y+1,{255,240,200,(unsigned char)(over?40:15)});

    Color tc=enabled?(over?accent:C_TEXT):C_TEXT_DIM;
    float tw=MeasureBody(label,sz);
    int lx=(int)(x+w/2-tw/2), ly=y+h/2-(int)(sz*0.5f);
    DrawBodyText(label, lx+1, ly+1, sz, {0,0,0,100});
    DrawBodyText(label, lx,   ly,   sz, tc);
    return over&&IsMouseButtonReleased(MOUSE_LEFT_BUTTON);
}

// ── Progress bar ─────────────────────────────────────────────────────────────
static void DrawProgressBar(int x, int y, int w, int h, float v){
    DrawRectangle(x,y,w,h,{20,16,10,200});
    DrawRectangleLines(x,y,w,h,C_BORDER);
    int f=(int)(v*(w-2));
    if(f>0){
        DrawRectangleGradientH(x+1,y+1,f,h-2,C_GOLD_DIM,C_GOLD);
        DrawRectangle(x+1,y+1,f,2,{255,240,180,60});
    }
}

// ── News / Changelog panel ────────────────────────────────────────────────────
static float gNewsScroll=0.0f;
static void DrawNewsPanel(int x, int y, int w, int h){
    DrawPanel(x,y,w,h);

    // Header
    const char* hdr="PATCH NOTES";
    float hfs=Sf(15.0f);
    float hw=MeasureBody(hdr,hfs);
    DrawBodyText(hdr, (int)(x+w/2-hw/2)+1, y+S(13), hfs, {0,0,0,140});
    DrawBodyText(hdr, (int)(x+w/2-hw/2),   y+S(12), hfs, C_GOLD);
    DrawGoldSep(x+S(16), y+S(34), w-S(32));

    int clipY=y+S(42), clipH=h-S(52);
    BeginScissorMode(x+S(8), clipY, w-S(16), clipH);
    int cy=y+S(46)-(int)gNewsScroll;
    for(int i=0;i<NUM_NEWS;i++){
        const NewsEntry& e=NEWS[i];
        float bfs=Sf(13.0f), dfs=Sf(12.0f);
        float vw=MeasureBody(e.version,bfs);
        int pillH=S(20), pillPad=S(16);
        DrawRectangle(x+S(10), cy, (int)vw+pillPad, pillH, C_GOLD_DIM);
        DrawRectangleLines(x+S(10),cy,(int)vw+pillPad,pillH,C_GOLD);
        DrawBodyText(e.version, x+S(18), cy+S(3), bfs, {240,220,160,255});
        DrawBodyText(e.date, x+S(10)+(int)vw+S(24), cy+S(4), dfs, C_TEXT_DIM);
        cy+=S(28);
        for(int j=0;j<e.nItems;j++){
            DrawRectangle(x+S(14), cy+S(7), S(4), S(4), C_GOLD_DIM);
            DrawBodyText(e.items[j], x+S(26), cy, bfs, C_TEXT);
            cy+=S(20);
        }
        cy+=S(14);
        if(i<NUM_NEWS-1){
            DrawLine(x+S(16),cy,x+w-S(16),cy,{C_BORDER.r,C_BORDER.g,C_BORDER.b,100});
            cy+=S(12);
        }
    }
    EndScissorMode();

    Vector2 mp=GetMousePosition();
    if(mp.x>=x&&mp.x<x+w&&mp.y>=y&&mp.y<y+h){
        gNewsScroll-=GetMouseWheelMove()*Sf(22.0f);
        gNewsScroll=Clamp(gNewsScroll,0.0f,(float)std::max(0,(cy-(y+h-S(52)))));
    }
    DrawRectangleGradientV(x+S(8),clipY,        w-S(16),S(22), {14,12,9,240},{14,12,9,0});
    DrawRectangleGradientV(x+S(8),y+h-S(30),    w-S(16),S(22), {14,12,9,0},  {14,12,9,240});
}

// ── Right panel: logo, status, play ──────────────────────────────────────────
static bool DrawRightPanel(int x, int y, int w, int h,
                           std::atomic<bool>& dlReq, float t){
    bool playClicked=false;
    DrawPanel(x,y,w,h,true);

    int ty=y+S(28);
    float pulse=0.5f+0.5f*sinf(t*1.8f);

    for(int r=30;r>0;r-=3){
        unsigned char a=(unsigned char)(pulse*8*(1.0f-(float)r/30.0f));
        DrawRectangle(x+w/2-r*3,ty-r/2,r*6,r+S(50),{C_GOLD.r,C_GOLD.g,C_GOLD.b,a});
    }

    const char* logo="DELVE";
    float lsz=Sf(54.0f);
    float lw=MeasureTitle(logo,lsz);
    int lx=(int)(x+w/2-lw/2);
    DrawTitleText(logo, lx+S(3), ty+S(4), lsz, {0,0,0,120});
    DrawTitleText(logo, lx+S(1), ty+S(2), lsz, {0,0,0,80});
    DrawTitleText(logo, lx, ty, lsz, {C_GOLD.r,C_GOLD.g,C_GOLD.b,(unsigned char)(80+pulse*50)});
    DrawTitleText(logo, lx, ty, lsz, C_GOLD_BRT);
    ty+=(int)(lsz)+S(8);

    const char* sub="Mine. Craft. Conquer.";
    float sfs=Sf(13.0f);
    float sw=MeasureBody(sub,sfs);
    DrawBodyText(sub,(int)(x+w/2-sw/2)+1,ty+1,sfs,{0,0,0,80});
    DrawBodyText(sub,(int)(x+w/2-sw/2),  ty,  sfs,C_TEXT_DIM);
    ty+=S(22);

    DrawGoldSep(x+S(20),ty,w-S(40)); ty+=S(16);

    // Version / status
    {
        float bfs=Sf(13.0f);
        char buf[64]; snprintf(buf,64,"Installed  v%s",CURRENT_VERSION);
        float bw=MeasureBody(buf,bfs);
        DrawBodyText(buf,(int)(x+w/2-bw/2),ty,bfs,C_TEXT_DIM);
        ty+=S(20);

        UpdateStatus st=gStatus.load();
        const char* stMsg=nullptr; Color stCol=C_TEXT_DIM;
        char dynBuf[80];

        if(st==US_CHECKING){
            int dots=((int)(t*3))%4;
            char dc[8]="";
            for(int i=0;i<dots;i++) strcat(dc,".");
            snprintf(dynBuf,80,"Checking for updates%s",dc);
            stMsg=dynBuf; stCol=C_TEXT_DIM;
        } else if(st==US_UP_TO_DATE){
            stMsg="Up to date  v"; stCol=C_GREEN;
        } else if(st==US_AVAILABLE){
            std::string lv; { std::lock_guard<std::mutex> lk(gMtx); lv=gLatestVersion; }
            snprintf(dynBuf,80,"Update available  v%s",lv.c_str());
            stMsg=dynBuf; stCol=C_BLUE;
        } else if(st==US_DOWNLOADING){
            int64_t got=gDlBytes.load(),tot=gDlTotal.load();
            if(tot>0) snprintf(dynBuf,80,"Downloading  %.1f / %.1f MB",got/1e6,tot/1e6);
            else      snprintf(dynBuf,80,"Downloading  %.1f MB",got/1e6);
            stMsg=dynBuf; stCol=C_BLUE;
        } else if(st==US_DONE){
            stMsg="Download complete!"; stCol=C_GREEN;
        } else if(st==US_ERROR){
            stMsg="Update check failed"; stCol=C_RED;
        } else if(st==US_OFFLINE){
            stMsg="Offline - playing locally"; stCol=C_TEXT_DIM;
        }
        if(stMsg){
            float mw=MeasureBody(stMsg,bfs);
            if(stCol.r!=C_TEXT_DIM.r||stCol.g!=C_TEXT_DIM.g)
                DrawBodyText(stMsg,(int)(x+w/2-mw/2),(int)ty,bfs,{stCol.r,stCol.g,stCol.b,40});
            DrawBodyText(stMsg,(int)(x+w/2-mw/2),(int)ty,bfs,stCol);
        }
        ty+=S(22);
    }

    UpdateStatus st=gStatus.load();
    if(st==US_DOWNLOADING){
        DrawProgressBar(x+S(16),ty,w-S(32),S(10),gDlProgress.load());
        ty+=S(18);
    }

    ty+=S(6);
    DrawGoldSep(x+S(20),ty,w-S(40)); ty+=S(16);

    int bpad=S(16), bw=w-bpad*2, bx2=x+bpad;
    if(st==US_AVAILABLE){
        if(DrawButton(bx2,ty,bw,S(36),"Download Update",C_BLUE,true,Sf(15.0f))){
            if(!dlReq.load()){ dlReq=true; gStatus=US_DOWNLOADING;
                               std::thread(DownloadThread).detach(); }
        }
        ty+=S(46);
        if(DrawButton(bx2,ty,bw,S(28),"Play Current Version",C_TEXT_DIM,true,Sf(13.0f)))
            playClicked=true;
        ty+=S(36);
    } else if(st==US_DONE){
        if(DrawButton(bx2,ty,bw,S(36),"Install Update",C_BLUE,true,Sf(15.0f))){
            std::string dlp; { std::lock_guard<std::mutex> lk(gMtx); dlp=gDlPath; }
            if(!dlp.empty())
                ShellExecuteA(nullptr,"runas",dlp.c_str(),nullptr,nullptr,SW_SHOWNORMAL);
            CloseWindow();
        }
        ty+=S(46);
    } else {
        bool canPlay=(st!=US_DOWNLOADING);
        if(DrawPlayButton(bx2,ty,bw,S(52),canPlay,t)) playClicked=true;
        ty+=S(62);
    }

    // Quit link
    {
        const char* ql="Quit";
        float qfs=Sf(13.0f);
        float qw=MeasureBody(ql,qfs);
        int qx=(int)(x+w/2-qw/2);
        Vector2 mp=GetMousePosition();
        bool hov=(mp.x>=qx&&mp.x<qx+(int)qw&&mp.y>=ty&&mp.y<ty+S(18));
        if(hov) DrawLine(qx,ty+S(15),qx+(int)qw,ty+S(15),{C_TEXT.r,C_TEXT.g,C_TEXT.b,120});
        DrawBodyText(ql,qx,ty,qfs,hov?C_TEXT:C_TEXT_DIM);
        if(hov&&IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) CloseWindow();
        ty+=S(20);
    }

    // Footer
    {
        char ft[64]; snprintf(ft,64,"Delve  v%s    (c) 2026",CURRENT_VERSION);
        float ffs=Sf(11.0f);
        float fw=MeasureBody(ft,ffs);
        DrawBodyText(ft,(int)(x+w/2-fw/2),y+h-S(18),ffs,C_TEXT_DIM);
    }
    return playClicked;
}

// ── Launch game ───────────────────────────────────────────────────────────────
static bool LaunchGame(){
    char exePath[MAX_PATH]={};
    GetModuleFileNameA(nullptr,exePath,MAX_PATH);
    char* sl=strrchr(exePath,'\\');
    if(sl) *(sl+1)='\0'; else exePath[0]='\0';
    std::string gamePath=std::string(exePath)+GAME_EXE;
    STARTUPINFOA si={}; si.cb=sizeof(si);
    PROCESS_INFORMATION pi={};
    return CreateProcessA(gamePath.c_str(),nullptr,nullptr,nullptr,
                          FALSE,0,nullptr,exePath,&si,&pi)!=0;
}

// ── Entry point ───────────────────────────────────────────────────────────────
int main(){
    SetConfigFlags(FLAG_WINDOW_UNDECORATED | FLAG_MSAA_4X_HINT);
    InitWindow(W,H,"Delve Launcher");
    SetTargetFPS(60);

    // Maximise to the monitor's work area (excludes taskbar)
    RECT wa; SystemParametersInfoA(SPI_GETWORKAREA,0,&wa,0);
    int mw=wa.right-wa.left, mh=wa.bottom-wa.top;

    // Scale factor relative to 1080p reference — everything stays proportional
    gScale = (float)mh / 1080.0f;

    SetWindowPosition(wa.left, wa.top);
    SetWindowSize(mw, mh);

    LoadFonts();

    Texture2D dummy={}; // unused — background is procedural now

    std::thread(CheckUpdateThread).detach();
    std::atomic<bool> dlReq{false};
    bool shouldPlay=false;

    Vector2 dragStart={0,0}; bool dragging=false;

    while(!WindowShouldClose()){
        float t=(float)GetTime();
        int SW=GetScreenWidth(), SH=GetScreenHeight();
        Vector2 mp=GetMousePosition();

        // Dragging
        if(IsMouseButtonPressed(MOUSE_LEFT_BUTTON)&&mp.y<S(36)){
            dragging=true; dragStart=mp;
        }
        if(dragging){
            if(IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
                Vector2 wp=GetWindowPosition();
                SetWindowPosition((int)(wp.x+mp.x-dragStart.x),
                                  (int)(wp.y+mp.y-dragStart.y));
                dragStart=mp;
            } else dragging=false;
        }

        if(shouldPlay){ if(LaunchGame()) break; shouldPlay=false; }

        BeginDrawing();

        // Background
        DrawBackground(t, SW, SH);

        // ── Custom title bar ───────────────────────────────────────────────
        int TB=S(36); // titlebar height
        DrawRectangleGradientV(0,0,SW,TB,{6,5,10,250},{6,5,10,200});
        DrawLine(0,TB,SW,TB,{C_BORDER.r,C_BORDER.g,C_BORDER.b,150});
        DrawLine(0,TB+1,SW,TB+1,{C_GOLD.r,C_GOLD.g,C_GOLD.b,30});

        const char* wt="Delve Launcher";
        float wtw=MeasureBody(wt,Sf(13.0f));
        DrawBodyText(wt,(int)(SW/2-wtw/2),S(11),Sf(13.0f),C_TEXT_DIM);

        bool xhov=(mp.x>=SW-S(36)&&mp.x<SW&&mp.y>=0&&mp.y<TB);
        if(xhov) DrawRectangle(SW-S(36),0,S(36),TB,{200,60,60,60});
        DrawBodyText("x",SW-S(22),S(11),Sf(14.0f),xhov?C_RED:C_TEXT_DIM);
        if(xhov&&IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) break;

        // ── Content panels ─────────────────────────────────────────────────
        const int PAD=S(12), TOP=TB+S(10);
        const int newsW=(int)(SW*0.60f), rightW=SW-newsW-PAD*3;
        DrawNewsPanel(PAD, TOP, newsW, SH-TOP-PAD);
        if(DrawRightPanel(PAD+newsW+PAD, TOP, rightW, SH-TOP-PAD, dlReq, t))
            shouldPlay=true;

        EndDrawing();
    }

    if(gFontsLoaded){
        UnloadFont(gFontTitle);
        UnloadFont(gFontBody);
    }
    CloseWindow();
    return 0;
}