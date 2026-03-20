#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"
#include "FastNoiseLite.h"
#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_set>
#include <cstring>

const int   CHUNK_SIZE           = 16;
const int   CHUNK_HEIGHT         = 512;
const int   SECTION_HEIGHT        = 16;   // vertical sub-mesh height
const int   NUM_SECTIONS          = CHUNK_HEIGHT / SECTION_HEIGHT; // 32
const float BLOCK_SIZE           = 1.0f;
const int   RENDER_DISTANCE      = 3;
const int   MAX_BUILDS_PER_FRAME = 4;


// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

static const char* WORLD_VS = R"glsl(
#version 330
in vec3 vertexPosition;
in vec2 vertexTexCoord;
in vec4 vertexColor;
uniform mat4 mvp;
out vec2 fragUV;
out vec4 fragLight;   // rgb=packed normal, a=skyFactor*variation
out vec3 fragWorldPos;
void main() {
    fragUV       = vertexTexCoord;
    fragLight    = vertexColor;
    fragWorldPos = vertexPosition;
    gl_Position  = mvp * vec4(vertexPosition, 1.0);
}
)glsl";

static const char* WORLD_FS = R"glsl(
#version 330
in vec2  fragUV;
in vec4  fragLight;
in vec3  fragWorldPos;
uniform sampler2D texture0;
uniform vec3  camPos;
uniform vec3  fogColor;
uniform float fogStart;
uniform float fogEnd;
uniform vec3  uSunColor;
uniform float uSunIntensity;
uniform vec3  uSkyColor;
uniform vec3  uSunDir;
// Point lights (torches)
uniform int   uLightCount;
uniform vec3  uLightPos[8];
uniform vec3  uLightColor[8];
out vec4 finalColor;
void main() {
    vec3  n        = normalize(fragLight.rgb * 2.0 - 1.0);
    float skyFact  = fragLight.a;

    float sunD     = max(dot(n, normalize(uSunDir)), 0.0);
    vec3  sunL     = uSunColor * uSunIntensity * sunD;

    float skyA     = (n.y * 0.5 + 0.5) * 0.6;
    vec3  skyL     = uSkyColor * skyA * skyFact;

    // Raised floor from 0.06→0.10 so caves aren't pitch-black without torches
    vec3  ambL     = vec3(0.10 + 0.05 * skyFact);

    // Point light contribution from torches (8 lights max — halves shader work vs 16)
    vec3 ptL = vec3(0.0);
    for(int i=0;i<uLightCount;i++){
        vec3  toL  = uLightPos[i] - fragWorldPos;
        float dist2= dot(toL,toL);
        // Reduced coefficient 0.06→0.04 extends useful torch range from ~8 to ~10 blocks
        float att  = 1.0 / (1.0 + dist2 * 0.04);
        float diff = max(dot(n, normalize(toL)), 0.0);
        float ambient = att * 0.30;  // slightly stronger soft ambient wrap
        ptL += uLightColor[i] * (diff * att + ambient);
    }

    vec3 col  = texture(texture0, fragUV).rgb * (sunL + skyL + ambL + ptL);
    float dist = length(fragWorldPos.xz - camPos.xz);
    float fogT = clamp((dist - fogStart) / (fogEnd - fogStart), 0.0, 1.0);
    col = mix(col, fogColor, fogT * fogT);
    finalColor = vec4(col, 1.0);
}
)glsl";

// Sky: purely vertical gradient + sun disc at the 3D sun direction
static const char* SKY_VS = R"glsl(
#version 330
in vec3 vertexPosition;
uniform mat4 mvp;
out vec3 skyDir;
void main() {
    skyDir = normalize(vertexPosition);
    vec4 pos = mvp * vec4(vertexPosition, 1.0);
    gl_Position = pos.xyww;
}
)glsl";

static const char* SKY_FS = R"glsl(
#version 330
in vec3 skyDir;
uniform vec3  sunDir;
uniform float uSunElev;   // raw sun elevation: -1=midnight, 0=horizon, 1=noon
out vec4 finalColor;
void main() {
    vec3  d   = normalize(skyDir);
    float h   = clamp(d.y, 0.0, 1.0);
    float elev = uSunElev;

    // Three sky palette states — night, golden hour, midday
    // Night
    vec3 zenithN  = vec3(0.00, 0.00, 0.05);
    vec3 midN     = vec3(0.00, 0.01, 0.08);
    vec3 highHorN = vec3(0.01, 0.01, 0.06);
    vec3 horizonN = vec3(0.01, 0.02, 0.07);
    // Golden hour (dawn / dusk) — warm orange/purple
    vec3 zenithG  = vec3(0.06, 0.06, 0.18);
    vec3 midG     = vec3(0.18, 0.10, 0.28);
    vec3 highHorG = vec3(0.55, 0.22, 0.18);
    vec3 horizonG = vec3(0.82, 0.38, 0.08);
    // Midday — clean blue
    vec3 zenithD  = vec3(0.08, 0.18, 0.55);
    vec3 midD     = vec3(0.18, 0.32, 0.68);
    vec3 highHorD = vec3(0.46, 0.58, 0.80);
    vec3 horizonD = vec3(0.72, 0.80, 0.90);

    // Blend weights: tGolden peaks at horizon (elev≈0), tDay grows toward noon
    float tGolden = smoothstep(-0.15, 0.0, elev) * (1.0 - smoothstep(0.15, 0.45, elev));
    float tDay    = smoothstep(0.10, 0.45, elev);

    vec3 zenith      = mix(mix(zenithN, zenithG, tGolden), zenithD, tDay);
    vec3 midBlue     = mix(mix(midN,    midG,    tGolden), midD,    tDay);
    vec3 highHorizon = mix(mix(highHorN,highHorG,tGolden), highHorD,tDay);
    vec3 horizon     = mix(mix(horizonN,horizonG,tGolden), horizonD,tDay);

    vec3 sky = mix(horizon,      highHorizon, smoothstep(0.00, 0.12, h));
    sky      = mix(sky,          midBlue,     smoothstep(0.08, 0.35, h));
    sky      = mix(sky,          zenith,      smoothstep(0.25, 1.00, h));

    // Horizon haze: warm orange at golden hour, pale blue at midday
    vec3  hazeCol = mix(vec3(0.70, 0.28, 0.04), vec3(0.65, 0.78, 0.92), tDay);
    float haze    = pow(1.0 - abs(d.y), 5.0);
    sky += hazeCol * haze * 0.30 * smoothstep(-0.12, 0.05, elev);

    // Sun disc + halo — color shifts orange at horizon, white at noon
    vec3  sun      = normalize(sunDir);
    float sd       = dot(d, sun);
    float disc     = smoothstep(0.9994, 0.9998, sd);
    float halo     = pow(max(sd, 0.0), 48.0) * 0.70;
    float glow     = pow(max(sd, 0.0), 10.0) * 0.25;
    vec3  sunDiscCol = mix(vec3(1.00, 0.80, 0.40), vec3(1.00, 0.97, 0.88), tDay);
    vec3  haloCol    = mix(vec3(1.00, 0.45, 0.05), vec3(1.00, 0.85, 0.55), tDay);
    vec3  glowCol    = mix(vec3(0.90, 0.25, 0.02), vec3(0.80, 0.88, 1.00), tDay);
    float aboveHorizon = smoothstep(-0.08, 0.05, elev);
    sky += sunDiscCol * disc;
    sky += haloCol * halo * aboveHorizon;
    sky += glowCol * glow * aboveHorizon;

    // Moon — cool white disc, visible only at night
    float nightFade = 1.0 - smoothstep(-0.10, 0.10, elev);
    vec3  moon  = -sun;
    float md    = dot(d, moon);
    float mDisc = smoothstep(0.9994, 0.9998, md);
    float mGlow = pow(max(md, 0.0), 28.0) * 0.10;
    sky += vec3(0.90, 0.93, 1.00) * mDisc * nightFade;
    sky += vec3(0.50, 0.60, 0.85) * mGlow * nightFade;

    finalColor = vec4(sky, 1.0);
}
)glsl";

// God rays: simple radial blur toward the sun's current screen position
// sunUV is updated every frame from the 3D sun projection
static const char* GODRAY_FS = R"glsl(
#version 330
in vec2 fragTexCoord;
uniform sampler2D texture0;
uniform vec2  sunUV;
uniform float sunVisible;   // 0 when sun is behind camera
out vec4 finalColor;

const int   SAMPLES  = 80;
const float DECAY    = 0.968;
const float DENSITY  = 0.55;
const float EXPOSURE = 0.072;
const float WEIGHT   = 0.95;

void main() {
    vec2 uv    = fragTexCoord;
    vec2 delta = (uv - sunUV) * (DENSITY / float(SAMPLES));

    float illum = 1.0;
    vec3  color = vec3(0.0);
    vec2  tc    = uv;

    for (int i = 0; i < SAMPLES; i++) {
        tc -= delta;
        vec3  s   = texture(texture0, clamp(tc, 0.001, 0.999)).rgb;
        float lum = dot(s, vec3(0.299, 0.587, 0.114));
        // Raise threshold so only the bright sun disc (not general sky) feeds rays
        color  += s * max(lum - 0.82, 0.0) * WEIGHT * illum;
        illum  *= DECAY;
    }

    color *= EXPOSURE * sunVisible;

    vec3 scene = texture(texture0, fragTexCoord).rgb;
    finalColor = vec4(scene + color * vec3(1.0, 0.60, 0.20), 1.0);
}
)glsl";

// ---------------------------------------------------------------------------
// Frustum culling helpers
// ---------------------------------------------------------------------------
struct FrustumPlane { float a,b,c,d; };

static void ExtractFrustum(Camera3D& cam, int SW, int SH, FrustumPlane planes[6]){
    float aspect = (float)SW / (float)SH;
    Matrix proj = MatrixPerspective(cam.fovy * DEG2RAD, aspect, 0.1f, 1200.0f);
    Matrix view = MatrixLookAt(cam.position, cam.target, cam.up);
    Matrix pv   = MatrixMultiply(view, proj);
    // Gribb-Hartmann extraction from combined matrix (column-major raymath convention)
    // Left
    planes[0]={pv.m3+pv.m0, pv.m7+pv.m4, pv.m11+pv.m8, pv.m15+pv.m12};
    // Right
    planes[1]={pv.m3-pv.m0, pv.m7-pv.m4, pv.m11-pv.m8, pv.m15-pv.m12};
    // Bottom
    planes[2]={pv.m3+pv.m1, pv.m7+pv.m5, pv.m11+pv.m9, pv.m15+pv.m13};
    // Top
    planes[3]={pv.m3-pv.m1, pv.m7-pv.m5, pv.m11-pv.m9, pv.m15-pv.m13};
    // Near
    planes[4]={pv.m3+pv.m2, pv.m7+pv.m6, pv.m11+pv.m10,pv.m15+pv.m14};
    // Far — skip for voxels (chunks at edge always partially visible)
    planes[5]={0,0,0,1}; // always pass
    for(int i=0;i<5;i++){
        float len=sqrtf(planes[i].a*planes[i].a+planes[i].b*planes[i].b+planes[i].c*planes[i].c);
        if(len>0){ planes[i].a/=len; planes[i].b/=len; planes[i].c/=len; planes[i].d/=len; }
    }
}

// Test AABB against frustum — returns true if visible (any part inside all planes)
static bool AABBInFrustum(FrustumPlane planes[6], Vector3 bmin, Vector3 bmax){
    for(int i=0;i<6;i++){
        // Positive vertex (most positive along plane normal)
        float px = planes[i].a>0 ? bmax.x : bmin.x;
        float py = planes[i].b>0 ? bmax.y : bmin.y;
        float pz = planes[i].c>0 ? bmax.z : bmin.z;
        if(planes[i].a*px + planes[i].b*py + planes[i].c*pz + planes[i].d < 0)
            return false; // entirely outside this plane
    }
    return true;
}


// ---------------------------------------------------------------------------
// Block types
// ---------------------------------------------------------------------------
enum BlockType : uint8_t {
    BLOCK_AIR=0,
    BLOCK_STONE, BLOCK_DIRT, BLOCK_GRASS,
    // Ores
    BLOCK_COAL, BLOCK_IRON, BLOCK_GOLD, BLOCK_DIAMOND,
    BLOCK_BEDROCK,
    BLOCK_TORCH,
    BLOCK_COUNT
};

// Seconds to mine each block type
float BlockHardness(BlockType t){
    switch(t){
        case BLOCK_GRASS:    return 0.4f;
        case BLOCK_DIRT:     return 0.5f;
        case BLOCK_STONE:    return 1.4f;
        case BLOCK_COAL:     return 1.6f;
        case BLOCK_IRON:     return 2.2f;
        case BLOCK_GOLD:     return 2.0f;
        case BLOCK_DIAMOND:  return 4.0f;
        case BLOCK_BEDROCK:  return 1e30f;
        case BLOCK_TORCH:    return 0.1f;  // mine instantly
        default:             return 1.0f;
    }
}

// Max stack size per block type
const char* BlockName(BlockType t){
    switch(t){
        case BLOCK_STONE:   return "Stone";
        case BLOCK_DIRT:    return "Dirt";
        case BLOCK_GRASS:   return "Grass";
        case BLOCK_COAL:    return "Coal Ore";
        case BLOCK_IRON:    return "Iron Ore";
        case BLOCK_GOLD:    return "Gold Ore";
        case BLOCK_DIAMOND: return "Diamond Ore";
        case BLOCK_BEDROCK: return "Bedrock";
        case BLOCK_TORCH:   return "Torch";
        default:            return "";
    }
}

int BlockMaxStack(BlockType t){
    // future: minerals->32, gems->8, tools->1
    // all current blocks are terrain, stack to 64
    switch(t){
        default: return 64;
    }
}

// Representative color for particles (matches atlas tint)
Color BlockColor(BlockType t){
    switch(t){
        case BLOCK_GRASS: return {60,140,45,255};
        case BLOCK_DIRT:  return {110,82,55,255};
        case BLOCK_STONE: return {110,108,108,255};
        default:          return {128,128,128,255};
    }
}

// Atlas layout: 7 solid block types × 6 faces = 42 tiles, each 64px wide
// STONE:0-5  DIRT:6-11  GRASS:12-17  COAL:18-23  IRON:24-29  GOLD:30-35  DIAMOND:36-41  BEDROCK:42-47
// Face indices: 0=top 1=bottom 2=south 3=north 4=east 5=west
const int ATLAS_TILES = 54; // +6 for torch

int GetTile(BlockType t, int face) {
    int base = 0;
    switch(t){
        case BLOCK_DIRT:    base=6;  break;
        case BLOCK_GRASS:   base=12; break;
        case BLOCK_COAL:    base=18; break;
        case BLOCK_IRON:    base=24; break;
        case BLOCK_GOLD:    base=30; break;
        case BLOCK_DIAMOND: base=36; break;
        case BLOCK_BEDROCK: base=42; break;
        case BLOCK_TORCH:   base=48; break;
        default: break;
    }
    return base + face;
}
void TileUV(int tile, float& u0, float& u1){
    u0=(float)tile /ATLAS_TILES;
    u1=(float)(tile+1)/ATLAS_TILES;
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------
static const Vector3 FACE_NORMAL[6] = {
    {0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,0,0},{-1,0,0}
};
float BlockVariation(int wx, int wy, int wz) {
    int h = wx*374761393 + wy*1234567 + wz*914729;
    h = (h^(h>>13))*1664525 + 1013904223;
    return 0.88f + ((float)((h>>8)&0xFF)/255.0f)*0.24f;
}

// Packs face data into vertex color for dynamic lighting in the shader:
//   RGB = face normal packed from -1..1 into 0..255
//   A   = skyFactor * blockVariation  (sky exposure × per-block noise)
// The shader unpacks the normal and computes sun diffuse against the live sunDir.
Color VertexLight(int face, int wx, int wy, int wz, float skyFactor) {
    Vector3 n   = FACE_NORMAL[face];
    float   var = BlockVariation(wx, wy, wz);
    // Pack normal: 0=−1, 128=0, 255=+1
    unsigned char r = (unsigned char)((n.x + 1.0f) * 0.5f * 255.0f);
    unsigned char g = (unsigned char)((n.y + 1.0f) * 0.5f * 255.0f);
    unsigned char b = (unsigned char)((n.z + 1.0f) * 0.5f * 255.0f);
    unsigned char a = (unsigned char)Clamp(skyFactor * var * 255.0f, 0.0f, 255.0f);
    return {r, g, b, a};
}

// ---------------------------------------------------------------------------
// Texture pack
// ---------------------------------------------------------------------------
// Each block has ONE png in textures/ using a cube-unfold cross layout:
//
//        col:  0      1      2      3
//   row 0:           [top  ]
//   row 1:  [west ][front][east ][back ]
//   row 2:           [bttm ]
//
// Image size: 256 × 192 px  (4 × 3 tiles of 64 × 64 each)
// Missing file → default is generated and saved so artists have a start.
// ---------------------------------------------------------------------------

static const int T = 64; // tile size in pixels

// Shared noise helper
static float TexNoise(int x,int y,int s){
    int h=x*1619+y*31337+s*3121;
    h=(h^(h>>13))*1664525+1013904223;
    return (float)((h>>8)&0xFF)/255.0f;
}

// Generate the default cross image for each block type (4T × 3T)
static Image GenCross_Stone(){
    Image img=GenImageColor(T*4,T*3,{0,0,0,0});
    // All 6 faces identical for stone
    int facePos[6][2]={{1,0},{1,2},{1,1},{3,1},{2,1},{0,1}};
    for(auto& fp:facePos){
        for(int py=0;py<T;py++) for(int px=0;px<T;px++){
            float n=TexNoise(px,py,3)*0.20f+TexNoise(px+7,py+3,9)*0.12f;
            float crack=(TexNoise(px,py,7)>0.88f)?0.75f:1.0f;
            unsigned char b=(unsigned char)Clamp((88+n*255*0.35f)*crack,0,255);
            ImageDrawPixel(&img,fp[0]*T+px,fp[1]*T+py,{b,b,(unsigned char)Clamp(b*1.05f,0,255),255});
        }
    }
    return img;
}
static Image GenCross_Dirt(){
    Image img=GenImageColor(T*4,T*3,{0,0,0,0});
    int facePos[6][2]={{1,0},{1,2},{1,1},{3,1},{2,1},{0,1}};
    for(auto& fp:facePos){
        for(int py=0;py<T;py++) for(int px=0;px<T;px++){
            float n=TexNoise(px,py,2)*0.25f;
            ImageDrawPixel(&img,fp[0]*T+px,fp[1]*T+py,{(unsigned char)(108+n*100),(unsigned char)(80+n*76),(unsigned char)(55+n*64),255});
        }
    }
    return img;
}
static Image GenCross_Grass(){
    Image img=GenImageColor(T*4,T*3,{0,0,0,0});
    // top face (1,0) — green
    for(int py=0;py<T;py++) for(int px=0;px<T;px++){
        float n=TexNoise(px,py,0)*0.22f;
        ImageDrawPixel(&img,T+px,py,{(unsigned char)(52+n*100),(unsigned char)(118+n*127),(unsigned char)(38+n*50),255});
    }
    // bottom face (1,2) — dirt
    for(int py=0;py<T;py++) for(int px=0;px<T;px++){
        float n=TexNoise(px,py,2)*0.25f;
        ImageDrawPixel(&img,T+px,T*2+py,{(unsigned char)(108+n*100),(unsigned char)(80+n*76),(unsigned char)(55+n*64),255});
    }
    // four side faces (front/east/back/west at cols 1,2,3,0 of row 1)
    int sideCols[]={1,2,3,0};
    for(int sc:sideCols){
        for(int py=0;py<T;py++) for(int px=0;px<T;px++){
            float n=TexNoise(px,py,1)*0.2f;
            float bl=Clamp((1.0f-(float)py/T)*3.0f,0.0f,1.0f);
            ImageDrawPixel(&img,sc*T+px,T+py,{
                (unsigned char)((68+n*40)*bl+(95+n*30)*(1-bl)),
                (unsigned char)((108+n*40)*bl+(72+n*25)*(1-bl)),
                (unsigned char)((48+n*20)*bl+(52+n*20)*(1-bl)),255});
        }
    }
    return img;
}

// Generate a stone-base cross with a colored vein overlay for ore blocks
static Image GenCross_Ore(Color veinColor, int seed){
    // Start with stone as base
    Image img = GenCross_Stone();
    auto rng=[](int x,int y,int s)->float{
        int h=x*1619+y*31337+s*3121;
        h=(h^(h>>13))*1664525+1013904223;
        return (float)((h>>8)&0xFF)/255.0f;
    };
    // Paint scattered vein pixels on every face
    int facePos[6][2]={{1,0},{1,2},{1,1},{3,1},{2,1},{0,1}};
    for(auto& fp:facePos){
        for(int py=0;py<T;py++) for(int px=0;px<T;px++){
            float n = rng(px,py,seed)*rng(px+3,py+5,seed+1);
            if(n > 0.55f){
                // blend vein colour over stone
                float str = (n-0.55f)*3.5f;
                Color cur; // read existing pixel
                // approximate — just overdraw with alpha blend manually
                unsigned char or_=(unsigned char)(veinColor.r*str + 88*(1-str));
                unsigned char og_=(unsigned char)(veinColor.g*str + 88*(1-str));
                unsigned char ob_=(unsigned char)(veinColor.b*str + 92*(1-str));
                ImageDrawPixel(&img, fp[0]*T+px, fp[1]*T+py, {or_,og_,ob_,255});
            }
        }
    }
    return img;
}
static Image GenCross_Bedrock(){
    const int T2=64;
    Image img=GenImageColor(T2*4,T2*3,{0,0,0,0});
    auto rng=[](int x,int y,int s)->float{
        int h=x*1619+y*31337+s*3121;
        h=(h^(h>>13))*1664525+1013904223;
        return (float)((h>>8)&0xFF)/255.0f;
    };
    int facePos[6][2]={{1,0},{1,2},{1,1},{3,1},{2,1},{0,1}};
    for(auto& fp:facePos){
        for(int py=0;py<T2;py++) for(int px=0;px<T2;px++){
            float n=rng(px,py,50)*0.18f + rng(px+3,py+7,51)*0.08f;
            // Dark grey-purple base with subtle cracks
            float crack=(rng(px,py,52)>0.92f)?0.6f:1.0f;
            unsigned char b=(unsigned char)Clamp((18+n*30)*crack,0,255);
            unsigned char r=(unsigned char)Clamp(b*0.95f,0,255);
            ImageDrawPixel(&img,fp[0]*T2+px,fp[1]*T2+py,{r,b,(unsigned char)Clamp(b*1.08f,0,255),255});
        }
    }
    return img;
}
static Image GenCross_Coal()    { return GenCross_Ore({30, 30, 30,255}, 10); }

static Image GenCross_Torch(){
    // Torch: transparent background, thin stick in center, flame on top
    Image img = GenImageColor(T*4, T*3, {0,0,0,0});
    // Draw on the 'front' face cell (col=1, row=1)
    // Stick: 6px wide centered, from bottom to 75% height
    for(int face=0;face<4;face++){
        int cellX=(face==0?1:face==1?2:face==2?3:0)*T;
        int cellY=T; // row 1
        // Background: dark transparent
        for(int y=0;y<T;y++) for(int x=0;x<T;x++)
            ImageDrawPixel(&img,cellX+x,cellY+y,{0,0,0,0});
        // Stick: warm brown, 6px wide center column
        int sx=T/2-3, sw=6;
        for(int y=T/4;y<T;y++) for(int x=sx;x<sx+sw;x++){
            float n=TexNoise(x,y,42+face)*0.15f;
            unsigned char r=(unsigned char)(120+n*60), g=(unsigned char)(72+n*40), b=(unsigned char)(30+n*20);
            ImageDrawPixel(&img,cellX+x,cellY+y,{r,g,b,255});
        }
        // Flame: orange glow, 10px wide, top quarter
        for(int y=0;y<T/4+4;y++) for(int x=T/2-5;x<T/2+5;x++){
            float t2=(float)y/(T/4+4);
            float n=TexNoise(x,y,99+face)*0.3f;
            unsigned char r=(unsigned char)(255);
            unsigned char g=(unsigned char)Clamp((1.0f-t2)*160+n*60,0,255);
            unsigned char b=(unsigned char)Clamp(n*30,0,255);
            unsigned char a=(unsigned char)Clamp((1.0f-t2*t2)*240,0,255);
            ImageDrawPixel(&img,cellX+x,cellY+y,{r,g,b,a});
        }
    }
    // Top face: small flame circle
    for(int y=0;y<T;y++) for(int x=0;x<T;x++){
        float dx=(float)(x-T/2)/T, dy=(float)(y-T/2)/T;
        float r2=dx*dx+dy*dy;
        if(r2<0.08f){
            float n=TexNoise(x,y,77)*0.3f;
            unsigned char g2=(unsigned char)Clamp((1.0f-r2/0.08f)*160+n*60,0,255);
            ImageDrawPixel(&img,T+x,0+y,{255,g2,0,(unsigned char)Clamp((1.0f-r2/0.08f)*230,0,255)});
        }
    }
    return img;
}
static Image GenCross_Iron()    { return GenCross_Ore({190,140, 90,255}, 20); }
static Image GenCross_Gold()    { return GenCross_Ore({230,195, 30,255}, 30); }
static Image GenCross_Diamond() { return GenCross_Ore({60, 220,220,255}, 40); }

struct BlockTexEntry {
    BlockType   type;
    const char* filename;
    int         atlasBase;
    Image(*generate)();
};
// Face→(col,row) in cross: top(1,0) bottom(1,2) south(1,1) north(3,1) east(2,1) west(0,1)
static const int CROSS_COL[6]={1,1,1,3,2,0};
static const int CROSS_ROW[6]={0,2,1,1,1,1};

static const BlockTexEntry BLOCK_TEX[] = {
    {BLOCK_STONE,    "textures/stone.png",   0,  GenCross_Stone   },
    {BLOCK_DIRT,     "textures/dirt.png",    6,  GenCross_Dirt    },
    {BLOCK_GRASS,    "textures/grass.png",   12, GenCross_Grass   },
    {BLOCK_COAL,     "textures/coal.png",    18, GenCross_Coal    },
    {BLOCK_IRON,     "textures/iron.png",    24, GenCross_Iron    },
    {BLOCK_GOLD,     "textures/gold.png",    30, GenCross_Gold    },
    {BLOCK_DIAMOND,  "textures/diamond.png", 36, GenCross_Diamond },
    {BLOCK_BEDROCK,  "textures/bedrock.png",  42, GenCross_Bedrock },
    {BLOCK_TORCH,    "textures/torch.png",    48, GenCross_Torch   },
};
static const int NUM_BLOCK_TEX = 9;

Texture2D BuildAtlas() {
    std::filesystem::create_directories("textures");

    // Atlas: ATLAS_TILES wide, 1 tile tall (18 × 64 = 1152 × 64)
    Image atlas = GenImageColor(T*ATLAS_TILES, T, {0,0,0,255}); // 42*64 = 2688px wide

    for(int b=0;b<NUM_BLOCK_TEX;b++){
        const BlockTexEntry& be = BLOCK_TEX[b];
        Image cross;
        if(std::filesystem::exists(be.filename)){
            cross = LoadImage(be.filename);
            // Resize to canonical 4T×3T if needed
            if(cross.width!=T*4||cross.height!=T*3)
                ImageResize(&cross,T*4,T*3);
        } else {
            cross = be.generate();
            ExportImage(cross, be.filename);
            TraceLog(LOG_INFO,"TEXTURES: Generated default %s",be.filename);
        }
        // Extract each face from the cross and write into the atlas
        for(int f=0;f<6;f++){
            int srcX=CROSS_COL[f]*T, srcY=CROSS_ROW[f]*T;
            int dstX=(be.atlasBase+f)*T;
            Rectangle src={(float)srcX,(float)srcY,(float)T,(float)T};
            Rectangle dst={(float)dstX,0,(float)T,(float)T};
            ImageDraw(&atlas, cross, src, dst, WHITE);
        }
        UnloadImage(cross);
    }

    // Write a README so artists know the layout
    if(!std::filesystem::exists("textures/README.txt")){
        FILE* f=fopen("textures/README.txt","w");
        if(f){
            fprintf(f,"Delve Texture Pack\n");
            fprintf(f,"==================\n\n");
            fprintf(f,"Each file is a 256 x 192 PNG (4 x 3 grid of 64x64 tiles).\n\n");
            fprintf(f,"Cube unfold layout:\n\n");
            fprintf(f,"       col:  0      1      2      3\n");
            fprintf(f,"  row 0:           [top  ]\n");
            fprintf(f,"  row 1:  [west ][front][east ][back ]\n");
            fprintf(f,"  row 2:           [bttm ]\n\n");
            fprintf(f,"Files:\n");
            fprintf(f,"  stone.png  — all 6 faces (uniform rock)\n");
            fprintf(f,"  dirt.png   — all 6 faces (uniform soil)\n");
            fprintf(f,"  grass.png  — top=grass, sides=grass-side, bottom=dirt\n\n");
            fprintf(f,"Tips:\n");
            fprintf(f,"  - Keep pixels sharp (no anti-aliasing)\n");
            fprintf(f,"  - Unused cross cells (corners) are ignored\n");
            fprintf(f,"  - Any image size works; game rescales to 256x192\n");
            fclose(f);
        }
    }

    Texture2D tex = LoadTextureFromImage(atlas);
    UnloadImage(atlas);
    SetTextureFilter(tex, TEXTURE_FILTER_POINT);
    SetTextureWrap(tex, TEXTURE_WRAP_CLAMP);
    return tex;
}


// ---------------------------------------------------------------------------
// GUI Texture
// ---------------------------------------------------------------------------
// gui.png (128×48 px) — replace to fully reskin the UI.
//
// Layout:
//   (0,0)  18×18 — slot_normal   (background when not selected)
//   (20,0) 18×18 — slot_selected (background when selected / hotbar highlight)
//   (40,0) 24×24 — panel_9slice  (stretched panel bg; border=3px)
//
// To reskin: edit textures/gui.png in any pixel-art editor.
// Keep the regions at the same positions; sizes must stay the same.
// ---------------------------------------------------------------------------

// GUI sprite sheet regions (source coords in gui.png)
struct GuiRegions {
    static constexpr Rectangle slotNormal   = {  0, 0, 18, 18 };
    static constexpr Rectangle slotSelected = { 20, 0, 18, 18 };
    static constexpr Rectangle panel        = { 40, 0, 24, 24 };
    static constexpr int       panelBorder  = 3;
};

static Image GenGuiImage() {
    // 128×48 pixel canvas
    Image img = GenImageColor(128, 48, {0,0,0,0});

    auto FillRect=[&](int x,int y,int w,int h,Color c){
        for(int py=y;py<y+h;py++) for(int px=x;px<x+w;px++)
            ImageDrawPixel(&img,px,py,c);
    };
    auto DrawBorder=[&](int x,int y,int w,int h,int thick,Color c){
        FillRect(x,      y,      w,    thick, c);  // top
        FillRect(x,      y+h-thick, w, thick, c);  // bottom
        FillRect(x,      y,      thick, h,    c);  // left
        FillRect(x+w-thick,y,    thick, h,    c);  // right
    };

    // ── slot_normal (0,0) 18×18 ──────────────────────────────────────────────
    FillRect(0,0,18,18,{28,24,18,210});     // dark fill
    DrawBorder(0,0,18,18,1,{75,68,52,255}); // outer border
    DrawBorder(1,1,16,16,1,{18,15,10,180}); // inner shadow

    // ── slot_selected (20,0) 18×18 ───────────────────────────────────────────
    FillRect(20,0,18,18,{58,46,22,230});      // warm highlight fill
    DrawBorder(20,0,18,18,1,{215,175,55,255});// gold outer border
    DrawBorder(21,1,16,16,1,{240,210,80,180});// bright inner rim
    // subtle inner glow corners
    for(int i=2;i<4;i++){
        ImageDrawPixel(&img,20+i,  1,   {255,230,100,80});
        ImageDrawPixel(&img,20+16+i-2,1,{255,230,100,80});
    }

    // ── panel_9slice (40,0) 24×24, border=3 ──────────────────────────────────
    int px=40,py=0,pw=24,ph=24,pb=3;
    // center fill
    FillRect(px+pb,py+pb,pw-pb*2,ph-pb*2,{14,12,9,230});
    // edges — slightly lighter than center
    FillRect(px+pb,  py,      pw-pb*2, pb, {22,19,14,240}); // top edge
    FillRect(px+pb,  py+ph-pb,pw-pb*2, pb, {10, 8, 5,240}); // bottom edge
    FillRect(px,     py+pb,   pb,      ph-pb*2,{22,19,14,240}); // left
    FillRect(px+pw-pb,py+pb,  pb,      ph-pb*2,{10, 8, 5,240}); // right
    // corners
    FillRect(px,      py,      pb,pb,{30,26,18,245}); // TL
    FillRect(px+pw-pb,py,      pb,pb,{22,19,12,245}); // TR
    FillRect(px,      py+ph-pb,pb,pb,{12,10, 6,245}); // BL
    FillRect(px+pw-pb,py+ph-pb,pb,pb,{10, 8, 4,245}); // BR
    // outer border line (1px inset)
    DrawBorder(px,py,pw,ph,1,{90,80,60,200});
    // inner highlight (1px inside border)
    FillRect(px+1,py+1,pw-2,1,{60,54,38,120}); // top highlight
    FillRect(px+1,py+1,1,ph-2,{60,54,38,120}); // left highlight

    return img;
}

Texture2D BuildGuiTex() {
    std::filesystem::create_directories("textures");
    const char* path = "textures/gui.png";
    Image img;
    if(std::filesystem::exists(path)){
        img = LoadImage(path);
        // Don't resize — regions are position-dependent; just validate minimums
        if(img.width<64||img.height<24){
            UnloadImage(img);
            img = GenGuiImage();
            TraceLog(LOG_WARNING,"TEXTURES: gui.png too small, using default");
        }
    } else {
        img = GenGuiImage();
        ExportImage(img, path);
        TraceLog(LOG_INFO,"TEXTURES: Generated default textures/gui.png");
    }
    Texture2D tex = LoadTextureFromImage(img);
    UnloadImage(img);
    SetTextureFilter(tex, TEXTURE_FILTER_POINT);
    return tex;
}

// Draw a 9-slice from guiTex stretched to fit dst
// src = source region in guiTex, border = corner size in source pixels
void DrawNineSlice(Texture2D guiTex, Rectangle src, int border, Rectangle dst){
    float b  = (float)border;
    float mw = src.width  - b*2;   // source middle width
    float mh = src.height - b*2;   // source middle height
    float dw = dst.width  - b*2;   // dest middle width
    float dh = dst.height - b*2;   // dest middle height
    float sx = src.x, sy = src.y;
    float dx = dst.x, dy = dst.y;

    // 9 pieces: TL T TR / L C R / BL B BR
    struct P { Rectangle s, d; };
    P pieces[9] = {
        {{sx,    sy,    b,  b }, {dx,       dy,       b,  b }},  // TL
        {{sx+b,  sy,    mw, b }, {dx+b,     dy,       dw, b }},  // T
        {{sx+b+mw,sy,   b,  b }, {dx+b+dw,  dy,       b,  b }},  // TR
        {{sx,    sy+b,  b,  mh}, {dx,       dy+b,     b,  dh}},  // L
        {{sx+b,  sy+b,  mw, mh}, {dx+b,     dy+b,     dw, dh}},  // C
        {{sx+b+mw,sy+b, b,  mh}, {dx+b+dw,  dy+b,     b,  dh}},  // R
        {{sx,    sy+b+mh,b, b }, {dx,       dy+b+dh,  b,  b }},  // BL
        {{sx+b,  sy+b+mh,mw,b }, {dx+b,     dy+b+dh,  dw, b }},  // B
        {{sx+b+mw,sy+b+mh,b,b }, {dx+b+dw,  dy+b+dh,  b,  b }},  // BR
    };
    for(auto& p:pieces)
        DrawTexturePro(guiTex, p.s, p.d, {0,0}, 0.0f, WHITE);
}

// ---------------------------------------------------------------------------
// Section flood-fill occlusion
// ---------------------------------------------------------------------------
// For a 16×16×16 section of blocks, computes which pairs of faces are
// connected through air. Stored as faceGraph[6]: bit j set in faceGraph[i]
// means face i can reach face j through air without passing through solid.
//
// Face indices: 0=+Y 1=-Y 2=+Z 3=-Z 4=+X 5=-X
// Uses iterative BFS with a fixed-size bitset (512 bits = 64 bytes for 4096 blocks).

// Direction vectors for each face
static const int FDIR[6][3] = {{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},{1,0,0},{-1,0,0}};
// Opposite face index
static const int FOPP[6] = {1,0,3,2,5,4};

static void ComputeSectionFaceGraph(
    const BlockType* blocks, // full chunk blocks array [x*CH*CS + y*CS + z]
    int chunkHeight, int chunkSize,
    int yMin,               // section start Y
    uint8_t faceGraph[6])   // output
{
    const int SH = SECTION_HEIGHT;
    const int CS = CHUNK_SIZE;

    // Quick all-air / all-solid check
    bool hasAir=false, hasSolid=false;
    for(int x=0;x<CS&&!(hasAir&&hasSolid);x++)
    for(int y=yMin;y<yMin+SH&&!(hasAir&&hasSolid);y++)
    for(int z=0;z<CS;z++){
        BlockType t=blocks[x*chunkHeight*CS+y*CS+z];
        if(t==BLOCK_AIR||t==BLOCK_TORCH) hasAir=true; else hasSolid=true;
    }
    if(!hasAir){ memset(faceGraph,0,6); return; }   // fully solid — nothing reachable
    if(!hasSolid){ memset(faceGraph,0x3F,6); return; } // fully air — everything reachable

    // BFS with a visited bitset (4096 bits = 64 uint64_t)
    uint64_t visited[64] = {};
    // Stack for BFS (max 4096 blocks)
    uint16_t stk[4096]; int top=0;

    auto idx=[&](int x,int y,int z)->int{ return x*SH*CS+(y-yMin)*CS+z; };
    auto isAir=[&](int x,int y,int z)->bool{
        BlockType t=blocks[x*chunkHeight*CS+y*CS+z];
        return t==BLOCK_AIR||t==BLOCK_TORCH;
    };

    // For each pair of faces (i,j), flood from face i, check if face j is hit
    // Optimization: flood ONCE per connected air region, record all touched faces
    // Use a global visited array reset per flood from each starting face

    for(int fi=0;fi<6;fi++){
        faceGraph[fi]=0;
        memset(visited,0,sizeof(visited));
        top=0;

        // Seed: all air blocks on face fi's border
        // Face 0(+Y): y=yMin+SH-1, Face 1(-Y): y=yMin
        // Face 2(+Z): z=CS-1,      Face 3(-Z): z=0
        // Face 4(+X): x=CS-1,      Face 5(-X): x=0
        for(int a=0;a<CS;a++) for(int b=0;b<CS;b++){
            int x,y,z;
            switch(fi){
                case 0: x=a;    y=yMin+SH-1; z=b; break;
                case 1: x=a;    y=yMin;       z=b; break;
                case 2: x=a;    y=yMin+b;     z=CS-1; break;
                case 3: x=a;    y=yMin+b;     z=0; break;
                case 4: x=CS-1; y=yMin+a;     z=b; break;
                default:x=0;    y=yMin+a;     z=b; break;
            }
            if(!isAir(x,y,z)) continue;
            int id=idx(x,y,z);
            if(visited[id>>6]&(1ULL<<(id&63))) continue;
            visited[id>>6]|=(1ULL<<(id&63));
            stk[top++]=(uint16_t)id;
            faceGraph[fi]|=(1<<fi); // face reaches itself
        }

        // BFS
        while(top>0){
            int id=stk[--top];
            int lx=id/(SH*CS), rem=id%(SH*CS), ly=yMin+rem/CS, lz=rem%CS;

            // Check which faces this block touches
            if(ly==yMin+SH-1) faceGraph[fi]|=(1<<0);
            if(ly==yMin)      faceGraph[fi]|=(1<<1);
            if(lz==CS-1)      faceGraph[fi]|=(1<<2);
            if(lz==0)         faceGraph[fi]|=(1<<3);
            if(lx==CS-1)      faceGraph[fi]|=(1<<4);
            if(lx==0)         faceGraph[fi]|=(1<<5);

            // Expand to 6 neighbours within section bounds
            const int nx[6]={lx,lx,lx,lx,lx+1,lx-1};
            const int ny[6]={ly+1,ly-1,ly,ly,ly,ly};
            const int nz[6]={lz,lz,lz+1,lz-1,lz,lz};
            for(int d=0;d<6;d++){
                int bx=nx[d],by=ny[d],bz=nz[d];
                if(bx<0||bx>=CS||by<yMin||by>=yMin+SH||bz<0||bz>=CS) continue;
                if(!isAir(bx,by,bz)) continue;
                int nid=idx(bx,by,bz);
                if(visited[nid>>6]&(1ULL<<(nid&63))) continue;
                visited[nid>>6]|=(1ULL<<(nid&63));
                stk[top++]=(uint16_t)nid;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Async mesh building data types + filler
// ---------------------------------------------------------------------------

// Snapshot passed to a worker thread — no live pointers, fully thread-safe
struct MeshJob {
    int cx, cz, sec;
    int yMin;
    float dist2 = 0.0f;
    // Live pointer to chunk blocks — chunk is pinned (meshJobsInFlight>0) until job completes
    // Worker only reads the section range; main thread won't write while secMeshInFlight=true
    const BlockType* liveBlocks = nullptr; // chunk->blocks.data()
    // Precomputed surfaceH (snapshot — cheap, only 1KB)
    int surfH[CHUNK_SIZE][CHUNK_SIZE];
    // Neighbour border slices — must snapshot since neighbours can be unloaded
    BlockType nbNX[SECTION_HEIGHT+2][CHUNK_SIZE];
    BlockType nbPX[SECTION_HEIGHT+2][CHUNK_SIZE];
    BlockType nbNZ[CHUNK_SIZE][SECTION_HEIGHT+2];
    BlockType nbPZ[CHUNK_SIZE][SECTION_HEIGHT+2];
    int snbNX[CHUNK_SIZE]; int snbPX[CHUNK_SIZE];
    int snbNZ[CHUNK_SIZE]; int snbPZ[CHUNK_SIZE];
};

// Filled vertex data returned by worker.
// Workers MemAlloc the buffers — main thread just assigns to Mesh and calls UploadMesh.
// This eliminates all allocation+memcpy from the main thread drain.
struct ReadyMesh {
    int            cx, cz, sec;
    // Raw MemAlloc'd buffers — ownership transferred to Mesh on drain
    float*          verts    = nullptr;
    float*          uvs      = nullptr;
    unsigned char*  cols     = nullptr;
    unsigned short* ids      = nullptr;
    int             vertCount= 0;  // number of vertices (fc*4)
    int             triCount = 0;  // number of triangles (fc*2)

    // Release all buffers (if not consumed by mesh — e.g. chunk unloaded)
    void Free(){ if(verts){MemFree(verts);verts=nullptr;}
                 if(uvs)  {MemFree(uvs);  uvs=nullptr;}
                 if(cols) {MemFree(cols);  cols=nullptr;}
                 if(ids)  {MemFree(ids);   ids=nullptr;}
                 vertCount=0; triCount=0; }
};

// Pure CPU vertex filler — no GPU calls, safe to run on any thread.
// Uses a local tmp vector for counting, then MemAlloc's exact-sized buffers.
// Main thread receives pre-allocated buffers — zero alloc/copy on drain.
static void FillMeshData(const MeshJob& job, ReadyMesh& out){
    out.Free();
    unsigned short vi=0;
    int yMin=job.yMin, yMax=yMin+SECTION_HEIGHT;

    auto GetB=[&](int x,int y,int z)->BlockType{
        if(y<0||y>=CHUNK_HEIGHT) return BLOCK_AIR;
        int yo=y-(job.yMin-1);
        if(x<0)           return (yo>=0&&yo<SECTION_HEIGHT+2)?job.nbNX[yo][z]:BLOCK_AIR;
        if(x>=CHUNK_SIZE) return (yo>=0&&yo<SECTION_HEIGHT+2)?job.nbPX[yo][z]:BLOCK_AIR;
        if(z<0)           return (yo>=0&&yo<SECTION_HEIGHT+2)?job.nbNZ[x][yo]:BLOCK_AIR;
        if(z>=CHUNK_SIZE) return (yo>=0&&yo<SECTION_HEIGHT+2)?job.nbPZ[x][yo]:BLOCK_AIR;
        // Own section — read directly from live chunk blocks (no copy)
        return job.liveBlocks[x*CHUNK_HEIGHT*CHUNK_SIZE + y*CHUNK_SIZE + z];
    };
    auto IsAirL=[&](int x,int y,int z)->bool{ BlockType bt=GetB(x,y,z); return bt==BLOCK_AIR||bt==BLOCK_TORCH; };

    auto ColSkyFactor=[&](int bx,int bz,int fromY)->float{
        int sh;
        if(bx<0)           sh=job.snbNX[bz<0?0:bz>=CHUNK_SIZE?CHUNK_SIZE-1:bz];
        else if(bx>=CHUNK_SIZE) sh=job.snbPX[bz<0?0:bz>=CHUNK_SIZE?CHUNK_SIZE-1:bz];
        else if(bz<0)      sh=job.snbNZ[bx];
        else if(bz>=CHUNK_SIZE) sh=job.snbPZ[bx];
        else sh=job.surfH[bx][bz];
        int solidAbove=std::max(0,std::min(8,sh-fromY));
        return Clamp(1.0f-solidAbove*0.35f,0.0f,1.0f);
    };

    // Single-pass: pre-allocate at worst case (all faces exposed), fill directly.
    // Avoids double-iteration — workers do one pass instead of two.
    // Worst case: 16×16×16 × 3 exposed faces average = 12,288 quads
    const int MAX_QUADS = CHUNK_SIZE*SECTION_HEIGHT*CHUNK_SIZE*3;
    out.verts    = (float*)         MemAlloc(MAX_QUADS*4*3*sizeof(float));
    out.uvs      = (float*)         MemAlloc(MAX_QUADS*4*2*sizeof(float));
    out.cols     = (unsigned char*) MemAlloc(MAX_QUADS*4*4*sizeof(unsigned char));
    out.ids      = (unsigned short*)MemAlloc(MAX_QUADS*2*3*sizeof(unsigned short));
    // vertCount/triCount set at end based on actual vi

    int vi3=0, vi2=0, ci=0, ii=0;
    auto Quad=[&](Vector3 a,Vector3 b,Vector3 c,Vector3 d,float u0,float u1,
                  Color c0,Color c1,Color c2,Color c3){
        out.verts[vi3+0]=a.x; out.verts[vi3+1]=a.y; out.verts[vi3+2]=a.z;
        out.verts[vi3+3]=b.x; out.verts[vi3+4]=b.y; out.verts[vi3+5]=b.z;
        out.verts[vi3+6]=c.x; out.verts[vi3+7]=c.y; out.verts[vi3+8]=c.z;
        out.verts[vi3+9]=d.x; out.verts[vi3+10]=d.y;out.verts[vi3+11]=d.z;
        vi3+=12;
        out.uvs[vi2+0]=u0; out.uvs[vi2+1]=1;
        out.uvs[vi2+2]=u1; out.uvs[vi2+3]=1;
        out.uvs[vi2+4]=u1; out.uvs[vi2+5]=0;
        out.uvs[vi2+6]=u0; out.uvs[vi2+7]=0;
        vi2+=8;
        Color cls[4]={c0,c1,c2,c3};
        for(auto& cl:cls){
            out.cols[ci++]=cl.r; out.cols[ci++]=cl.g;
            out.cols[ci++]=cl.b; out.cols[ci++]=cl.a;
        }
        out.ids[ii+0]=vi; out.ids[ii+1]=vi+1; out.ids[ii+2]=vi+2;
        out.ids[ii+3]=vi; out.ids[ii+4]=vi+2; out.ids[ii+5]=vi+3;
        ii+=6; vi+=4;
    };

    for(int x=0;x<CHUNK_SIZE;x++) for(int y=yMin;y<yMax;y++) for(int z=0;z<CHUNK_SIZE;z++){
        BlockType t=GetB(x,y,z); if(t==BLOCK_AIR||t==BLOCK_TORCH) continue;
        float bwx=(job.cx*CHUNK_SIZE+x)*BLOCK_SIZE;
        float bwy=y*BLOCK_SIZE, bwz=(job.cz*CHUNK_SIZE+z)*BLOCK_SIZE, s=BLOCK_SIZE;
        int iwx=job.cx*CHUNK_SIZE+x, iwz=job.cz*CHUNK_SIZE+z;

        float sfOwn=ColSkyFactor(x,z,y+1),  sfPZ=ColSkyFactor(x,z+1,y+1);
        float sfNZ=ColSkyFactor(x,z-1,y+1), sfPX=ColSkyFactor(x+1,z,y+1);
        float sfNX=ColSkyFactor(x-1,z,y+1);

        auto CalcAO=[&](int bx,int by,int bz,int s1x,int s1y,int s1z,int s2x,int s2y,int s2z)->float{
            bool a=!IsAirL(bx+s1x,by+s1y,bz+s1z);
            bool b=!IsAirL(bx+s2x,by+s2y,bz+s2z);
            bool c=!IsAirL(bx+s1x+s2x,by+s1y+s2y,bz+s1z+s2z);
            if(a&&b) return 0.0f;
            return 1.0f-(float)(a+b+c)*0.12f;
        };
        auto VCol=[](Color base,float ao)->Color{
            return {base.r,base.g,base.b,(unsigned char)Clamp((float)base.a*ao,0.0f,255.0f)};
        };

        float u0,u1; Color bc,v0,v1,v2,v3;
        if(IsAirL(x,y+1,z)){ TileUV(GetTile(t,0),u0,u1); bc=VertexLight(0,iwx,y,iwz,sfOwn);
            v0=VCol(bc,CalcAO(x,y,z,-1,+1,0,0,+1,+1)); v1=VCol(bc,CalcAO(x,y,z,+1,+1,0,0,+1,+1));
            v2=VCol(bc,CalcAO(x,y,z,+1,+1,0,0,+1,-1)); v3=VCol(bc,CalcAO(x,y,z,-1,+1,0,0,+1,-1));
            Quad({bwx,bwy+s,bwz+s},{bwx+s,bwy+s,bwz+s},{bwx+s,bwy+s,bwz},{bwx,bwy+s,bwz},u0,u1,v0,v1,v2,v3);}
        if(IsAirL(x,y-1,z)){ TileUV(GetTile(t,1),u0,u1); bc=VertexLight(1,iwx,y,iwz,sfOwn);
            v0=VCol(bc,CalcAO(x,y,z,-1,-1,0,0,-1,-1)); v1=VCol(bc,CalcAO(x,y,z,+1,-1,0,0,-1,-1));
            v2=VCol(bc,CalcAO(x,y,z,+1,-1,0,0,-1,+1)); v3=VCol(bc,CalcAO(x,y,z,-1,-1,0,0,-1,+1));
            Quad({bwx,bwy,bwz},{bwx+s,bwy,bwz},{bwx+s,bwy,bwz+s},{bwx,bwy,bwz+s},u0,u1,v0,v1,v2,v3);}
        if(IsAirL(x,y,z+1)){ TileUV(GetTile(t,2),u0,u1); bc=VertexLight(2,iwx,y,iwz,sfPZ);
            v0=VCol(bc,CalcAO(x,y,z,-1,0,+1,0,-1,+1)); v1=VCol(bc,CalcAO(x,y,z,+1,0,+1,0,-1,+1));
            v2=VCol(bc,CalcAO(x,y,z,+1,0,+1,0,+1,+1)); v3=VCol(bc,CalcAO(x,y,z,-1,0,+1,0,+1,+1));
            Quad({bwx,bwy,bwz+s},{bwx+s,bwy,bwz+s},{bwx+s,bwy+s,bwz+s},{bwx,bwy+s,bwz+s},u0,u1,v0,v1,v2,v3);}
        if(IsAirL(x,y,z-1)){ TileUV(GetTile(t,3),u0,u1); bc=VertexLight(3,iwx,y,iwz,sfNZ);
            v0=VCol(bc,CalcAO(x,y,z,+1,0,-1,0,-1,-1)); v1=VCol(bc,CalcAO(x,y,z,-1,0,-1,0,-1,-1));
            v2=VCol(bc,CalcAO(x,y,z,-1,0,-1,0,+1,-1)); v3=VCol(bc,CalcAO(x,y,z,+1,0,-1,0,+1,-1));
            Quad({bwx+s,bwy,bwz},{bwx,bwy,bwz},{bwx,bwy+s,bwz},{bwx+s,bwy+s,bwz},u0,u1,v0,v1,v2,v3);}
        if(IsAirL(x+1,y,z)){ TileUV(GetTile(t,4),u0,u1); bc=VertexLight(4,iwx,y,iwz,sfPX);
            v0=VCol(bc,CalcAO(x,y,z,+1,0,+1,+1,-1,0)); v1=VCol(bc,CalcAO(x,y,z,+1,0,-1,+1,-1,0));
            v2=VCol(bc,CalcAO(x,y,z,+1,0,-1,+1,+1,0)); v3=VCol(bc,CalcAO(x,y,z,+1,0,+1,+1,+1,0));
            Quad({bwx+s,bwy,bwz+s},{bwx+s,bwy,bwz},{bwx+s,bwy+s,bwz},{bwx+s,bwy+s,bwz+s},u0,u1,v0,v1,v2,v3);}
        if(IsAirL(x-1,y,z)){ TileUV(GetTile(t,5),u0,u1); bc=VertexLight(5,iwx,y,iwz,sfNX);
            v0=VCol(bc,CalcAO(x,y,z,-1,0,-1,-1,-1,0)); v1=VCol(bc,CalcAO(x,y,z,-1,0,+1,-1,-1,0));
            v2=VCol(bc,CalcAO(x,y,z,-1,0,+1,-1,+1,0)); v3=VCol(bc,CalcAO(x,y,z,-1,0,-1,-1,+1,0));
            Quad({bwx,bwy,bwz},{bwx,bwy,bwz+s},{bwx,bwy+s,bwz+s},{bwx,bwy+s,bwz},u0,u1,v0,v1,v2,v3);}
    }
    // Set actual counts — vi is quad count, buffers may be over-allocated but that's fine
    if(vi==0){ out.Free(); return; }
    out.vertCount = vi*4;
    out.triCount  = vi*2;
    // Trim to actual size (avoids wasting VRAM on GPU upload)
    out.verts = (float*)        MemRealloc(out.verts, vi*4*3*sizeof(float));
    out.uvs   = (float*)        MemRealloc(out.uvs,   vi*4*2*sizeof(float));
    out.cols  = (unsigned char*)MemRealloc(out.cols,  vi*4*4*sizeof(unsigned char));
    out.ids   = (unsigned short*)MemRealloc(out.ids,  vi*2*3*sizeof(unsigned short));
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------
struct Chunk {
    // Blocks stored on the heap — flat [x*CH*CS + y*CS + z]
    // NEVER declare Chunk as a local variable with old 3D array (stack overflow!)
    std::vector<BlockType> blocks;
    int chunkX=0,chunkZ=0;

    // One mesh per SECTION_HEIGHT-block vertical section — no Model needed
    // DrawMesh(mesh, material, transform) bypasses Model entirely
    Mesh  meshes[NUM_SECTIONS]  = {};
    bool  secDirty[NUM_SECTIONS];
    bool  secMeshInFlight[NUM_SECTIONS];
    bool    secAllSolid[NUM_SECTIONS];
    uint8_t secFaceGraph[NUM_SECTIONS][6]; // flood-fill occlusion: faceGraph[i] = bitmask of faces reachable from face i
    int     meshJobsInFlight = 0;
    int     occSlot = -1;  // index into World::occlusionVis array  // # jobs holding a pointer to our blocks — prevent deletion

    int surfaceH[CHUNK_SIZE][CHUNK_SIZE] = {}; // highest solid Y per column, baked at Generate()

    Chunk(){
        blocks.assign(CHUNK_SIZE*CHUNK_HEIGHT*CHUNK_SIZE, BLOCK_AIR);
        for(int i=0;i<NUM_SECTIONS;i++){ secDirty[i]=true; secMeshInFlight[i]=false; secAllSolid[i]=false; }
        memset(secFaceGraph, 0x3F, sizeof(secFaceGraph)); // init all-connected (safe default)
        memset(surfaceH, 0, sizeof(surfaceH));
    }

    inline BlockType Get(int x,int y,int z) const {
        // Bounds check — out-of-range treated as AIR (used by AO/lighting at chunk edges)
        if(x<0||x>=CHUNK_SIZE||y<0||y>=CHUNK_HEIGHT||z<0||z>=CHUNK_SIZE) return BLOCK_AIR;
        return blocks[x*CHUNK_HEIGHT*CHUNK_SIZE + y*CHUNK_SIZE + z];
    }
    inline void Set(int x,int y,int z,BlockType t){
        blocks[x*CHUNK_HEIGHT*CHUNK_SIZE + y*CHUNK_SIZE + z]=t;
        int s=y/SECTION_HEIGHT;
        secDirty[s]=true;
        secAllSolid[s]=false;
        memset(secFaceGraph[s], 0x3F, 6); // reopen all connections until recomputed
    }
    // Mark all sections dirty (used after Generate or neighbour load)
    void MarkAllDirty(){
        for(int i=0;i<NUM_SECTIONS;i++) secDirty[i]=true;
        // NOTE: do NOT reset secMeshInFlight here — in-flight jobs still need to complete
    }
    bool AnyDirty() const { for(int i=0;i<NUM_SECTIONS;i++) if(secDirty[i]) return true; return false; }

    // Only dirty sections that are NOT all-solid (solid sections can never have seam faces exposed)
    void MarkBorderDirty(){
        for(int i=0;i<NUM_SECTIONS;i++)
            if(!secAllSolid[i]) secDirty[i]=true;
    }

    void Generate(FastNoiseLite& noise, FastNoiseLite& caveNoise, FastNoiseLite& oreNoise,
                  std::unordered_map<int,BlockType>* savedMods){
        for(int x=0;x<CHUNK_SIZE;x++) for(int z=0;z<CHUNK_SIZE;z++){
            int wx=chunkX*CHUNK_SIZE+x, wz=chunkZ*CHUNK_SIZE+z;
            float n=noise.GetNoise((float)wx,(float)wz);
            int h=(int)(n*12.0f)+440;
            surfaceH[x][z] = h; // bake terrain height for O(1) sky factor lookup

            for(int y=0;y<CHUNK_HEIGHT;y++){
                BlockType base;
                if(y<2){ Set(x,y,z,BLOCK_BEDROCK); continue; }
                if(y>h)         base=BLOCK_AIR;
                else if(y==h)   base=BLOCK_GRASS;
                else if(y>=h-3) base=BLOCK_DIRT;
                else             base=BLOCK_STONE;

                // Cave carving — threshold grows quadratically with depth
                if(base!=BLOCK_AIR && y < h-2){
                    float cv1=caveNoise.GetNoise((float)wx,(float)y,(float)wz);
                    float cv2=caveNoise.GetNoise((float)wx+37.5f,(float)y*1.3f,(float)wz+19.1f);
                    float depthT=1.0f-(float)y/(float)(h-2);
                    depthT=depthT*depthT;
                    if(cv1*cv1+cv2*cv2 < 0.006f+depthT*0.025f) base=BLOCK_AIR; // rarer but wider
                }

                // Ore veins
                if(base==BLOCK_STONE){
                    float ov =oreNoise.GetNoise((float)wx,(float)y,(float)wz);
                    float ov2=oreNoise.GetNoise((float)wx*1.7f+100,(float)y*1.7f,(float)wz*1.7f+100);
                    if     (y<=380 && ov>0.55f && ov2>0.10f) base=BLOCK_COAL;
                    else if(y<=300 && ov>0.62f && ov2>0.15f) base=BLOCK_IRON;
                    else if(y<=180 && ov>0.70f && ov2>0.20f) base=BLOCK_GOLD;
                    else if(y<=80  && ov>0.76f && ov2>0.25f) base=BLOCK_DIAMOND;
                }
                Set(x,y,z,base);
            }
        }
        if(savedMods){
            for(auto& [idx,t]:*savedMods){
                int lx=idx/(CHUNK_HEIGHT*CHUNK_SIZE);
                int ly=(idx/CHUNK_SIZE)%CHUNK_HEIGHT;
                int lz=idx%CHUNK_SIZE;
                Set(lx,ly,lz,t);
            }
        }
        // Bake final surfaceH from actual block data (handles savedMods overrides)
        for(int x=0;x<CHUNK_SIZE;x++) for(int z=0;z<CHUNK_SIZE;z++){
            int top=0;
            for(int y=CHUNK_HEIGHT-1;y>=0;y--)
                if(Get(x,y,z)!=BLOCK_AIR){ top=y; break; }
            surfaceH[x][z]=top;
        }
        // Mark sections where every block is solid (no faces can be exposed internally)
        for(int s=0;s<NUM_SECTIONS;s++){
            int yMin=s*SECTION_HEIGHT, yMax=yMin+SECTION_HEIGHT;
            bool allSolid=true;
            for(int x=0;x<CHUNK_SIZE&&allSolid;x++)
            for(int y=yMin;y<yMax&&allSolid;y++)
            for(int z=0;z<CHUNK_SIZE&&allSolid;z++)
                if(Get(x,y,z)==BLOCK_AIR) allSolid=false;
            secAllSolid[s]=allSolid;
        }
        MarkAllDirty();
    }

    static bool IsTransparent(BlockType t){ return t==BLOCK_AIR||t==BLOCK_TORCH; }
    bool IsAir(int x,int y,int z,Chunk* nx,Chunk* px,Chunk* nz,Chunk* pz){
        if(y<0||y>=CHUNK_HEIGHT) return true;
        if(x<0)            return nx ? IsTransparent(nx->Get(CHUNK_SIZE-1,y,z)) : true;
        if(x>=CHUNK_SIZE)  return px ? IsTransparent(px->Get(0,y,z))           : true;
        if(z<0)            return nz ? IsTransparent(nz->Get(x,y,CHUNK_SIZE-1)): true;
        if(z>=CHUNK_SIZE)  return pz ? IsTransparent(pz->Get(x,y,0))           : true;
        return IsTransparent(Get(x,y,z));
    }

    // Build mesh for one 16-block vertical section. Safe with unsigned short indices
    // since a 16×16×16 section can have at most 16384 quads (well within 65535).
    void BuildSection(int sec, Chunk* cnx,Chunk* cpx,Chunk* cnz,Chunk* cpz,
                      Shader shader,Texture2D atlas){
        int yMin=sec*SECTION_HEIGHT, yMax=yMin+SECTION_HEIGHT;

        std::vector<float> verts,uvs;
        std::vector<unsigned char> cols;
        std::vector<unsigned short> ids;
        unsigned short vi=0;

        auto Quad=[&](Vector3 a,Vector3 b,Vector3 c,Vector3 d,float u0,float u1,
                      Color c0,Color c1,Color c2,Color c3){
            verts.insert(verts.end(),{a.x,a.y,a.z,b.x,b.y,b.z,c.x,c.y,c.z,d.x,d.y,d.z});
            uvs.insert(uvs.end(),{u0,1,u1,1,u1,0,u0,0});
            for(auto& cl:{c0,c1,c2,c3})
                cols.insert(cols.end(),{cl.r,cl.g,cl.b,cl.a});
            ids.insert(ids.end(),{vi,(unsigned short)(vi+1),(unsigned short)(vi+2),
                                   vi,(unsigned short)(vi+2),(unsigned short)(vi+3)});
            vi+=4;
        };

        for(int x=0;x<CHUNK_SIZE;x++) for(int y=yMin;y<yMax;y++) for(int z=0;z<CHUNK_SIZE;z++){
            BlockType t=Get(x,y,z); if(t==BLOCK_AIR||t==BLOCK_TORCH) continue;
            float bwx=(chunkX*CHUNK_SIZE+x)*BLOCK_SIZE;
            float bwy=y*BLOCK_SIZE;
            float bwz=(chunkZ*CHUNK_SIZE+z)*BLOCK_SIZE;
            float s=BLOCK_SIZE;
            int iwx=chunkX*CHUNK_SIZE+x, iwz=chunkZ*CHUNK_SIZE+z;

            // Sky factor: scan up to 64 blocks above (capped for performance)
            // O(1) sky factor: use precomputed surfaceH instead of scanning 8 rows
            auto ColSkyFactor=[&](int bx,int bz,int fromY)->float{
                if(bx<0||bx>=CHUNK_SIZE||bz<0||bz>=CHUNK_SIZE) return 1.0f;
                int solidAbove = std::max(0, std::min(8, surfaceH[bx][bz] - fromY));
                return Clamp(1.0f - solidAbove * 0.35f, 0.0f, 1.0f);
            };
            float sfOwn=ColSkyFactor(x,  z,  y+1);
            float sfPZ =ColSkyFactor(x,  z+1,y+1);
            float sfNZ =ColSkyFactor(x,  z-1,y+1);
            float sfPX =ColSkyFactor(x+1,z,  y+1);
            float sfNX =ColSkyFactor(x-1,z,  y+1);

            auto SolidAt=[&](int bx,int by,int bz)->bool{
                return !IsAir(bx,by,bz,cnx,cpx,cnz,cpz);
            };
            auto CalcAO=[&](int bx,int by,int bz,
                            int s1x,int s1y,int s1z,
                            int s2x,int s2y,int s2z)->float{
                bool a=SolidAt(bx+s1x,by+s1y,bz+s1z);
                bool b=SolidAt(bx+s2x,by+s2y,bz+s2z);
                bool c=SolidAt(bx+s1x+s2x,by+s1y+s2y,bz+s1z+s2z);
                if(a&&b) return 0.0f;
                return 1.0f-(float)(a+b+c)*0.12f;
            };
            auto VCol=[](Color base,float ao)->Color{
                return {base.r,base.g,base.b,
                        (unsigned char)Clamp((float)base.a*ao,0.0f,255.0f)};
            };

            float u0,u1; Color bc,v0,v1,v2,v3;
            if(IsAir(x,y+1,z,cnx,cpx,cnz,cpz)){
                TileUV(GetTile(t,0),u0,u1); bc=VertexLight(0,iwx,y,iwz,sfOwn);
                v0=VCol(bc,CalcAO(x,y,z,-1,+1,0,0,+1,+1));
                v1=VCol(bc,CalcAO(x,y,z,+1,+1,0,0,+1,+1));
                v2=VCol(bc,CalcAO(x,y,z,+1,+1,0,0,+1,-1));
                v3=VCol(bc,CalcAO(x,y,z,-1,+1,0,0,+1,-1));
                Quad({bwx,bwy+s,bwz+s},{bwx+s,bwy+s,bwz+s},{bwx+s,bwy+s,bwz},{bwx,bwy+s,bwz},u0,u1,v0,v1,v2,v3);
            }
            if(IsAir(x,y-1,z,cnx,cpx,cnz,cpz)){
                TileUV(GetTile(t,1),u0,u1); bc=VertexLight(1,iwx,y,iwz,sfOwn);
                v0=VCol(bc,CalcAO(x,y,z,-1,-1,0,0,-1,-1));
                v1=VCol(bc,CalcAO(x,y,z,+1,-1,0,0,-1,-1));
                v2=VCol(bc,CalcAO(x,y,z,+1,-1,0,0,-1,+1));
                v3=VCol(bc,CalcAO(x,y,z,-1,-1,0,0,-1,+1));
                Quad({bwx,bwy,bwz},{bwx+s,bwy,bwz},{bwx+s,bwy,bwz+s},{bwx,bwy,bwz+s},u0,u1,v0,v1,v2,v3);
            }
            if(IsAir(x,y,z+1,cnx,cpx,cnz,cpz)){
                TileUV(GetTile(t,2),u0,u1); bc=VertexLight(2,iwx,y,iwz,sfPZ);
                v0=VCol(bc,CalcAO(x,y,z,-1,0,+1,0,-1,+1));
                v1=VCol(bc,CalcAO(x,y,z,+1,0,+1,0,-1,+1));
                v2=VCol(bc,CalcAO(x,y,z,+1,0,+1,0,+1,+1));
                v3=VCol(bc,CalcAO(x,y,z,-1,0,+1,0,+1,+1));
                Quad({bwx,bwy,bwz+s},{bwx+s,bwy,bwz+s},{bwx+s,bwy+s,bwz+s},{bwx,bwy+s,bwz+s},u0,u1,v0,v1,v2,v3);
            }
            if(IsAir(x,y,z-1,cnx,cpx,cnz,cpz)){
                TileUV(GetTile(t,3),u0,u1); bc=VertexLight(3,iwx,y,iwz,sfNZ);
                v0=VCol(bc,CalcAO(x,y,z,+1,0,-1,0,-1,-1));
                v1=VCol(bc,CalcAO(x,y,z,-1,0,-1,0,-1,-1));
                v2=VCol(bc,CalcAO(x,y,z,-1,0,-1,0,+1,-1));
                v3=VCol(bc,CalcAO(x,y,z,+1,0,-1,0,+1,-1));
                Quad({bwx+s,bwy,bwz},{bwx,bwy,bwz},{bwx,bwy+s,bwz},{bwx+s,bwy+s,bwz},u0,u1,v0,v1,v2,v3);
            }
            if(IsAir(x+1,y,z,cnx,cpx,cnz,cpz)){
                TileUV(GetTile(t,4),u0,u1); bc=VertexLight(4,iwx,y,iwz,sfPX);
                v0=VCol(bc,CalcAO(x,y,z,+1,0,+1,+1,-1,0));
                v1=VCol(bc,CalcAO(x,y,z,+1,0,-1,+1,-1,0));
                v2=VCol(bc,CalcAO(x,y,z,+1,0,-1,+1,+1,0));
                v3=VCol(bc,CalcAO(x,y,z,+1,0,+1,+1,+1,0));
                Quad({bwx+s,bwy,bwz+s},{bwx+s,bwy,bwz},{bwx+s,bwy+s,bwz},{bwx+s,bwy+s,bwz+s},u0,u1,v0,v1,v2,v3);
            }
            if(IsAir(x-1,y,z,cnx,cpx,cnz,cpz)){
                TileUV(GetTile(t,5),u0,u1); bc=VertexLight(5,iwx,y,iwz,sfNX);
                v0=VCol(bc,CalcAO(x,y,z,-1,0,-1,-1,-1,0));
                v1=VCol(bc,CalcAO(x,y,z,-1,0,+1,-1,-1,0));
                v2=VCol(bc,CalcAO(x,y,z,-1,0,+1,-1,+1,0));
                v3=VCol(bc,CalcAO(x,y,z,-1,0,-1,-1,+1,0));
                Quad({bwx,bwy,bwz},{bwx,bwy,bwz+s},{bwx,bwy+s,bwz+s},{bwx,bwy+s,bwz},u0,u1,v0,v1,v2,v3);
            }
        }

        // Upload — free old mesh first
        if(meshes[sec].vertexCount>0){ UnloadMesh(meshes[sec]); meshes[sec]={}; }
        if(verts.empty()){ secDirty[sec]=false; return; }

        int fc=vi/4;
        meshes[sec].vertexCount  =fc*4;
        meshes[sec].triangleCount=fc*2;
        meshes[sec].vertices =(float*)         MemAlloc(fc*4*3*sizeof(float));
        meshes[sec].texcoords=(float*)         MemAlloc(fc*4*2*sizeof(float));
        meshes[sec].colors   =(unsigned char*) MemAlloc(fc*4*4*sizeof(unsigned char));
        meshes[sec].indices  =(unsigned short*)MemAlloc(fc*2*3*sizeof(unsigned short));
        for(int i=0;i<(int)verts.size();i++) meshes[sec].vertices[i] =verts[i];
        for(int i=0;i<(int)uvs.size();  i++) meshes[sec].texcoords[i]=uvs[i];
        for(int i=0;i<(int)cols.size(); i++) meshes[sec].colors[i]   =cols[i];
        for(int i=0;i<(int)ids.size();  i++) meshes[sec].indices[i]  =ids[i];
        UploadMesh(&meshes[sec],false);
        // No LoadModelFromMesh — caller uses DrawMesh with a shared Material
        secDirty[sec]=false;
    }

    // Draw is now called with an external Material — avoids Model overhead
    void Draw(Chunk* cnx,Chunk* cpx,Chunk* cnz,Chunk* cpz,Shader shader,Texture2D atlas,Material& mat){
        for(int s=0;s<NUM_SECTIONS;s++){
            if(secDirty[s]) BuildSection(s,cnx,cpx,cnz,cpz,shader,atlas);
            if(meshes[s].vertexCount>0) DrawMesh(meshes[s],mat,MatrixIdentity());
        }
    }
    void Unload(){
        for(int s=0;s<NUM_SECTIONS;s++)
            if(meshes[s].vertexCount>0){ UnloadMesh(meshes[s]); meshes[s]={}; }
    }
};

// RayHit struct
struct RayHit {
    bool  hit=false;
    int   wx=0,wy=0,wz=0;
    int   nx=0,ny=0,nz=0;
};

// ---------------------------------------------------------------------------
// Particle system
// ---------------------------------------------------------------------------
struct Particle {
    Vector3 pos, vel;
    Color   col;
    float   life, maxLife;
    float   size;
};

struct ParticleSystem {
    std::vector<Particle> particles;

    void Spawn(Vector3 origin, BlockType t, int count=14){
        Color base = BlockColor(t);
        for(int i=0;i<count;i++){
            float rx = ((float)GetRandomValue(-100,100)/100.0f);
            float ry = ((float)GetRandomValue(20,100)/100.0f);
            float rz = ((float)GetRandomValue(-100,100)/100.0f);
            float spd = (float)GetRandomValue(3,7);
            Particle p;
            p.pos     = {origin.x+0.5f, origin.y+0.5f, origin.z+0.5f};
            p.vel     = Vector3Scale(Vector3Normalize({rx,ry,rz}), spd);
            int var   = GetRandomValue(-18,18);
            p.col     = {
                (unsigned char)Clamp(base.r+var,0,255),
                (unsigned char)Clamp(base.g+var,0,255),
                (unsigned char)Clamp(base.b+var,0,255),
                255
            };
            p.maxLife = (float)GetRandomValue(35,65)/100.0f;
            p.life    = p.maxLife;
            p.size    = (float)GetRandomValue(4,10)/100.0f;
            particles.push_back(p);
        }
    }

    void Update(float dt){
        const float GRAVITY = -16.0f;
        for(auto& p:particles){
            p.vel.y += GRAVITY * dt;
            p.pos    = Vector3Add(p.pos, Vector3Scale(p.vel, dt));
            p.life  -= dt;
            p.col.a  = (unsigned char)(255.0f * Clamp(p.life/p.maxLife, 0.0f, 1.0f));
        }
        particles.erase(
            std::remove_if(particles.begin(), particles.end(),
                [](const Particle& p){ return p.life <= 0; }),
            particles.end());
    }

    void Draw(Camera3D& cam){
        for(auto& p:particles){
            Vector3 right = Vector3Normalize(Vector3CrossProduct(
                Vector3Subtract(cam.position, p.pos), {0,1,0}));
            Vector3 up = {0,1,0};
            float   s  = p.size;
            Vector3 a = Vector3Add(Vector3Add(p.pos, Vector3Scale(right,-s)), Vector3Scale(up,-s));
            Vector3 b = Vector3Add(Vector3Add(p.pos, Vector3Scale(right, s)), Vector3Scale(up,-s));
            Vector3 c = Vector3Add(Vector3Add(p.pos, Vector3Scale(right, s)), Vector3Scale(up, s));
            Vector3 d = Vector3Add(Vector3Add(p.pos, Vector3Scale(right,-s)), Vector3Scale(up, s));
            DrawTriangle3D(a, b, c, p.col);
            DrawTriangle3D(a, c, d, p.col);
        }
    }
};

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------
const int HOTBAR_SLOTS = 9;
const int INV_ROWS     = 3;
const int TOTAL_SLOTS  = HOTBAR_SLOTS + INV_ROWS * HOTBAR_SLOTS; // 36

struct ItemStack {
    BlockType type  = BLOCK_AIR;
    int       count = 0;
};

struct Inventory {
    ItemStack slots[TOTAL_SLOTS];  // 0-8 = hotbar, 9-35 = main grid
    int       selected  = 0;
    bool      open      = false;
    ItemStack held;               // item currently dragged by cursor

    bool Add(BlockType t){
        int mx = BlockMaxStack(t);
        for(int i=0;i<HOTBAR_SLOTS;i++)
            if(slots[i].type==t && slots[i].count<mx){ slots[i].count++; return true; }
        for(int i=HOTBAR_SLOTS;i<TOTAL_SLOTS;i++)
            if(slots[i].type==t && slots[i].count<mx){ slots[i].count++; return true; }
        for(int i=0;i<TOTAL_SLOTS;i++)
            if(slots[i].type==BLOCK_AIR){ slots[i].type=t; slots[i].count=1; return true; }
        return false;
    }

    bool Consume(int slot){
        if(slots[slot].count<=0) return false;
        slots[slot].count--;
        if(slots[slot].count==0) slots[slot].type=BLOCK_AIR;
        return true;
    }

    // Called when user clicks on slot index i while inventory is open
    void ClickSlot(int i, bool rightClick){
        if(i<0||i>=TOTAL_SLOTS) return;
        ItemStack& s = slots[i];

        if(held.type==BLOCK_AIR){
            // Nothing held — pick up from slot
            if(s.type==BLOCK_AIR) return;
            if(rightClick){
                // Pick up half (ceil)
                int take = (s.count+1)/2;
                held.type  = s.type;
                held.count = take;
                s.count   -= take;
                if(s.count==0) s.type=BLOCK_AIR;
            } else {
                // Pick up whole stack
                held = s;
                s    = {};
            }
        } else {
            // Holding something
            if(s.type==BLOCK_AIR){
                // Empty slot — place
                if(rightClick){
                    s.type=held.type; s.count=1;
                    held.count--;
                    if(held.count==0) held={};
                } else {
                    s=held; held={};
                }
            } else if(s.type==held.type){
                // Same type — stack as many as fit
                int mx  = BlockMaxStack(held.type);
                int fit = mx - s.count;
                if(fit<=0) return;
                if(rightClick){
                    s.count++;
                    held.count--;
                    if(held.count==0) held={};
                } else {
                    int move = (held.count<fit)?held.count:fit;
                    s.count   += move;
                    held.count -= move;
                    if(held.count==0) held={};
                }
            } else {
                // Different type — swap (only on LMB)
                if(!rightClick){
                    ItemStack tmp=s; s=held; held=tmp;
                }
            }
        }
    }

    // Returns which slot index is under pixel (mx,my), or -1
    int SlotAt(int mx, int my, int SW, int SH) const {
        const int SZ=50, PAD=4;
        const int rowW = HOTBAR_SLOTS*(SZ+PAD)-PAD;
        int hotX = SW/2 - rowW/2;
        int hotY = SH   - SZ - 14;

        // Hotbar
        for(int i=0;i<HOTBAR_SLOTS;i++){
            int x=hotX+i*(SZ+PAD);
            if(mx>=x&&mx<x+SZ&&my>=hotY&&my<hotY+SZ) return i;
        }
        if(!open) return -1;

        int panW = rowW+24;
        int panH = INV_ROWS*(SZ+PAD)+36;
        int panX = SW/2 - panW/2;
        int panY = hotY - panH - 8;
        int gridX= panX+12;
        int gridY= panY+32;

        // Main grid
        for(int row=0;row<INV_ROWS;row++){
            for(int col=0;col<HOTBAR_SLOTS;col++){
                int x=gridX+col*(SZ+PAD), y=gridY+row*(SZ+PAD);
                if(mx>=x&&mx<x+SZ&&my>=y&&my<y+SZ)
                    return HOTBAR_SLOTS+row*HOTBAR_SLOTS+col;
            }
        }
        return -1;
    }

    void HandleInput(bool& cursorLocked){
        if(IsKeyPressed(KEY_E)){
            open=!open;
            if(open){
                // Drop held item back if closing mid-drag
                EnableCursor();
                cursorLocked=false;
            } else {
                // Put held item back in first available slot
                if(held.type!=BLOCK_AIR){ Add(held.type); held={}; }
                DisableCursor();
                cursorLocked=true;
            }
        }
        if(!open){
            int w=(int)GetMouseWheelMove();
            if(w!=0) selected=(selected-w+HOTBAR_SLOTS)%HOTBAR_SLOTS;
            for(int i=0;i<HOTBAR_SLOTS;i++)
                if(IsKeyPressed(KEY_ONE+i)) selected=i;
        }
    }

    void HandleClick(int SW, int SH){
        if(!open) return;
        Vector2 mp = GetMousePosition();
        int mx=(int)mp.x, my=(int)mp.y;
        if(IsMouseButtonPressed(MOUSE_LEFT_BUTTON)){
            int idx=SlotAt(mx,my,SW,SH);
            if(idx>=0) ClickSlot(idx,false);
            // Click outside panel while holding → return to inventory
            else if(held.type!=BLOCK_AIR){ Add(held.type); held={}; }
        }
        if(IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)){
            int idx=SlotAt(mx,my,SW,SH);
            if(idx>=0) ClickSlot(idx,true);
        }
    }

    // Draw a single slot at pixel (x,y), size SZ, using atlas for icon
    void DrawSlot(int x, int y, int SZ, int slotIdx, bool sel, Texture2D atlas, Texture2D guiTex) const {
        // Draw slot background from gui.png sprite sheet
        Rectangle slotSrc = sel ? GuiRegions::slotSelected : GuiRegions::slotNormal;
        DrawTexturePro(guiTex, slotSrc, {(float)x,(float)y,(float)SZ,(float)SZ}, {0,0}, 0.0f, WHITE);

        if(slots[slotIdx].type==BLOCK_AIR) return;

        // Show the south/front face (face index 2) of the block
        BlockType t = slots[slotIdx].type;
        int tile = GetTile(t, 2); // face 2 = south/front
        const int ATLAS_W=T*ATLAS_TILES, ATLAS_H=T, TILE_W=T;
        Rectangle src = {(float)(tile*TILE_W), 0, (float)TILE_W, (float)ATLAS_H};
        int pad=5;
        Rectangle dst = {(float)(x+pad),(float)(y+pad),(float)(SZ-pad*2),(float)(SZ-pad*2)};
        DrawTexturePro(atlas, src, dst, {0,0}, 0.0f, WHITE);

        // Count badge (only show if >1)
        if(slots[slotIdx].count>1){
            char buf[8]; snprintf(buf,8,"%d",slots[slotIdx].count);
            int tw=MeasureText(buf,12);
            // shadow
            DrawText(buf, x+SZ-3-tw+1, y+SZ-15+1, 12, {0,0,0,200});
            DrawText(buf, x+SZ-3-tw,   y+SZ-15,   12, {230,215,175,255});
        }
        // Full indicator dot (at max stack)
        if(slots[slotIdx].count >= BlockMaxStack(t)){
            DrawCircle(x+SZ-7, y+7, 3, {80,200,80,220});
        }
    }

    void Draw(int SW, int SH, Texture2D atlas, Texture2D guiTex){
        const int SZ=50, PAD=4;
        const int rowW = HOTBAR_SLOTS*(SZ+PAD)-PAD;
        int hotX = SW/2 - rowW/2;
        int hotY = SH   - SZ - 14;

        // ── Hotbar (always visible) ──────────────────────────────────────────
        DrawNineSlice(guiTex, GuiRegions::panel, GuiRegions::panelBorder,
            {(float)(hotX-6),(float)(hotY-6),(float)(rowW+12),(float)(SZ+12)});
        for(int i=0;i<HOTBAR_SLOTS;i++){
            DrawSlot(hotX+i*(SZ+PAD), hotY, SZ, i, (i==selected), atlas, guiTex);
        }
        // selected item name
        if(slots[selected].type!=BLOCK_AIR){
            const char* nm=BlockName(slots[selected].type);
            int tw=MeasureText(nm,15);
            DrawText(nm, SW/2-tw/2+1, hotY-23+1, 15, {0,0,0,160});
            DrawText(nm, SW/2-tw/2,   hotY-23,   15, {220,210,180,230});
        }

        if(!open) return;

        // ── Full inventory panel ─────────────────────────────────────────────
        int panW = rowW+24;
        int panH = INV_ROWS*(SZ+PAD)+36; // 3 rows + title/hint padding
        int panX = SW/2 - panW/2;
        int panY = hotY - panH - 8;

        // Panel background — 9-slice from gui.png
        DrawNineSlice(guiTex, GuiRegions::panel, GuiRegions::panelBorder,
            {(float)panX,(float)panY,(float)panW,(float)panH});

        // Title
        const char* title="Inventory";
        DrawText(title, panX+panW/2-MeasureText(title,16)/2, panY+8, 16, {200,185,140,255});

        // Main grid rows (slots 9-35)
        int gridX = panX+12;
        int gridY = panY+32;
        for(int row=0;row<INV_ROWS;row++){
            for(int col=0;col<HOTBAR_SLOTS;col++){
                int idx = HOTBAR_SLOTS + row*HOTBAR_SLOTS + col;
                DrawSlot(gridX+col*(SZ+PAD), gridY+row*(SZ+PAD), SZ, idx, false, atlas, guiTex);
            }
        }

        // Key hint at bottom of panel
        DrawText("[E] close  LMB pick/place  RMB half", panX+6, panY+panH-16, 11, {120,110,90,180});
    }

    // Draw item currently held by cursor (call after Draw, in BeginDrawing)
    void DrawHeld(Texture2D atlas) const {
        if(held.type==BLOCK_AIR) return;
        Vector2 mp=GetMousePosition();
        const int SZ=44;
        int x=(int)mp.x-SZ/2, y=(int)mp.y-SZ/2;
        // icon
        int tile=GetTile(held.type,2); // south face
        Rectangle src={(float)(tile*T),0,(float)T,(float)T};
        Rectangle dst={(float)x,(float)y,(float)SZ,(float)SZ};
        DrawTexturePro(atlas,src,dst,{0,0},0.0f,WHITE);
        // count
        if(held.count>1){
            char buf[8]; snprintf(buf,8,"%d",held.count);
            int tw=MeasureText(buf,12);
            DrawText(buf,x+SZ-tw,y+SZ-14,12,{0,0,0,200});
            DrawText(buf,x+SZ-tw-1,y+SZ-15,12,{230,215,175,255});
        }
    }
};


// ---------------------------------------------------------------------------
// World
// ---------------------------------------------------------------------------
struct World {
    std::unordered_map<int64_t,Chunk> chunks;
    // Persistent block modifications: chunkKey → (packedLocalIdx → BlockType)
    std::unordered_map<int64_t, std::unordered_map<int,BlockType>> mods;
    FastNoiseLite noise;        // surface terrain height
    FastNoiseLite caveNoise;    // 3D cave carving
    FastNoiseLite oreNoise;     // ore vein placement
    int lastCX=99999,lastCZ=99999;

    Shader    worldShader, skyShader, godRayShader;
    Texture2D atlas;
    Texture2D guiTex;
    Model     skyModel;

    int locCamPos, locFogColor, locFogStart, locFogEnd;
    int locSkyDun, locSunElevLoc;
    int locSunUV, locSunVisible;
    int locSunColor, locSunIntensity, locSkyColor, locSunDir;
    int locLightCount, locLightPos, locLightColor;
    Material sectionMat;   // shared material for all terrain sections — set once
    Texture2D flameTex;
    Model     torchModel;
    bool      torchModelLoaded = false;
    struct TorchEntry { float x,y,z; };
    std::vector<TorchEntry> visibleTorches;      // filtered each frame from cache
    std::vector<TorchEntry> torchCache;          // ALL torch positions — rebuilt on change
    bool                    torchCacheDirty = true;
    std::vector<Mesh>       meshDeleteQueue; // deferred UnloadMesh — avoid mid-render GPU stall
    // Sorted chunk list cache — rebuilt only when chunks change or player crosses boundary
    // Occlusion: flat visibility array indexed by chunkSlot*NUM_SECTIONS+sec
    // chunkSlot is a small integer assigned when chunk loads (max 64 active chunks)
    static const int MAX_CHUNK_SLOTS = 128; // must exceed (RENDER_DISTANCE+2)^2 = 81 max chunks
    int64_t slotToKey[MAX_CHUNK_SLOTS];   // chunkKey at each slot (for slot recycling)
    Chunk*  slotToChunk[MAX_CHUNK_SLOTS];  // direct pointer — eliminates Get_byKey per BFS node
    int     nextFreeSlot = 0;
    // Each Chunk gets a slot index stored inside it
    // visibility[slot*NUM_SECTIONS+sec] = incomingFaceMask (0=not visited)
    uint8_t occlusionVis[MAX_CHUNK_SLOTS*NUM_SECTIONS];
    // Pre-allocated BFS queues — never reallocate
    struct BFSNode { int16_t slot; uint8_t s; uint8_t inFaces; };
    std::vector<BFSNode> bfsQ, bfsQ2;
    std::vector<Chunk*>     sortedChunksCache;
    bool                    sortedDirty   = true;
    int                     lastSortCX    = 99999;
    int                     lastSortCZ    = 99999;
    float dayTime  = 0.25f;
    Vector3 sunDir = {1.0f, 0.18f, 0.6f};
    float depthFade = 0.0f;

    // ── Thread pool for async chunk generation ────────────────────────────
    struct GenJob { int cx,cz; std::unordered_map<int,BlockType> savedMods; };
    struct ReadyChunk {
        int cx,cz;
        std::vector<BlockType> blocks;
        int  surfH[CHUNK_SIZE][CHUNK_SIZE];
        bool    secAllSolid[NUM_SECTIONS];
        uint8_t secFaceGraph[NUM_SECTIONS][6]; // occlusion face connectivity
    };
    std::vector<std::thread>       workers;
    std::queue<GenJob>             pendingJobs;
    std::mutex                     pendingMutex;
    std::condition_variable        pendingCV;
    std::queue<ReadyChunk>         readyChunks;
    std::mutex                     readyMutex;
    std::unordered_set<int64_t>    inFlight;   // keys queued or being generated
    std::atomic<bool>              stopWorkers{false};

    // ── Separate queue for async mesh building ────────────────────────────
    // Priority queue for mesh jobs — closest sections processed first
    // This ensures the player's immediate surroundings mesh before distant chunks
    struct MeshJobCmp {
        bool operator()(const std::unique_ptr<MeshJob>& a, const std::unique_ptr<MeshJob>& b) const {
            // lower priority value = further away = processed last
            // We store negative dist2 so smallest dist2 = highest priority
            return a->dist2 > b->dist2;
        }
    };
    std::priority_queue<std::unique_ptr<MeshJob>,
                        std::vector<std::unique_ptr<MeshJob>>,
                        MeshJobCmp> meshJobs;
    std::mutex                     meshJobMutex;
    std::condition_variable        meshJobCV;   // dedicated CV for mesh workers only
    std::queue<ReadyMesh>          meshReady;
    std::mutex                     meshReadyMutex;

    World(){
        noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        noise.SetFrequency(0.07f); noise.SetSeed(1337);

        // Cave noise — 3D Perlin, low frequency for large tunnels
        caveNoise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        caveNoise.SetFrequency(0.025f); caveNoise.SetSeed(9182); // lower freq = bigger tunnels

        // Ore noise — higher frequency for tight veins
        oreNoise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        oreNoise.SetFrequency(0.18f); oreNoise.SetSeed(4455);
    }

    void Init(){
        worldShader=LoadShaderFromMemory(WORLD_VS,WORLD_FS);
        atlas=BuildAtlas();
        guiTex=BuildGuiTex();
        locCamPos      =GetShaderLocation(worldShader,"camPos");
        locFogColor    =GetShaderLocation(worldShader,"fogColor");
        locFogStart    =GetShaderLocation(worldShader,"fogStart");
        locFogEnd      =GetShaderLocation(worldShader,"fogEnd");
        locSunColor    =GetShaderLocation(worldShader,"uSunColor");
        locSunIntensity=GetShaderLocation(worldShader,"uSunIntensity");
        locSkyColor    =GetShaderLocation(worldShader,"uSkyColor");
        locSunDir      =GetShaderLocation(worldShader,"uSunDir");
        locLightCount  =GetShaderLocation(worldShader,"uLightCount");
        locLightPos    =GetShaderLocation(worldShader,"uLightPos");
        locLightColor  =GetShaderLocation(worldShader,"uLightColor");
        // Shared terrain material — assigned to all section meshes at draw time
        sectionMat = LoadMaterialDefault();
        sectionMat.shader = worldShader;
        SetMaterialTexture(&sectionMat, MATERIAL_MAP_DIFFUSE, atlas);

        // Generate flame billboard texture (32×48: wide at bottom, tapers to tip)
        {
            const int FW=32, FH=48;
            Image fi=GenImageColor(FW,FH,{0,0,0,0});
            for(int y=0;y<FH;y++) for(int x=0;x<FW;x++){
                float cx2=(float)(x-FW/2)/(FW/2);   // -1..1
                float cy =(float)(FH-1-y)/FH;        // 0=bottom, 1=top
                // Flame shape: ellipse that tapers
                float hw=0.55f*(1.0f-cy*cy*0.8f);    // half-width narrows toward tip
                if(fabsf(cx2)>hw) continue;
                float edge=1.0f-fabsf(cx2)/hw;       // 0 at edge, 1 at centre
                float tip =1.0f-cy;                   // bright at bottom, dim at tip
                float intensity=edge*tip*0.9f+0.1f;
                // Colour: white core → orange → red at tip
                unsigned char r=255;
                unsigned char g=(unsigned char)Clamp((1.0f-cy*0.9f)*200*intensity,0,255);
                unsigned char b=(unsigned char)Clamp((1.0f-cy*2.0f)*60*intensity,0,255);
                unsigned char a=(unsigned char)Clamp(intensity*230,0,255);
                ImageDrawPixel(&fi,x,y,{r,g,b,a});
            }
            flameTex=LoadTextureFromImage(fi);
            UnloadImage(fi);
            SetTextureFilter(flameTex,TEXTURE_FILTER_BILINEAR);
        }
        // Load torch model — user can replace models/torch.obj with any Blender export
        {
            std::filesystem::create_directories("models");
            if(!std::filesystem::exists("models/torch.obj")){
                // Write minimal default OBJ (a square-prism stick)
                FILE* f=fopen("models/torch.obj","w");
                if(f){
                    fputs(
                        "# Torch — auto-generated by Delve\n"
                        "# Replace with your own Blender OBJ (1 unit = 1 block)\n"
                        "# Origin at block corner; game centres it at draw time\n"
                        "v 0.44 0.05 0.44\nv 0.56 0.05 0.44\nv 0.56 0.05 0.56\nv 0.44 0.05 0.56\n"
                        "v 0.44 0.65 0.44\nv 0.56 0.65 0.44\nv 0.56 0.65 0.56\nv 0.44 0.65 0.56\n"
                        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
                        "vn 0 1 0\nvn 0 -1 0\nvn 0 0 1\nvn 0 0 -1\nvn 1 0 0\nvn -1 0 0\n"
                        "f 5/4/1 6/3/1 7/2/1\nf 5/4/1 7/2/1 8/1/1\n"
                        "f 1/1/2 4/2/2 3/3/2\nf 1/1/2 3/3/2 2/4/2\n"
                        "f 4/1/3 8/2/3 7/3/3\nf 4/1/3 7/3/3 3/4/3\n"
                        "f 2/1/4 6/2/4 5/3/4\nf 2/1/4 5/3/4 1/4/4\n"
                        "f 3/1/5 7/2/5 6/3/5\nf 3/1/5 6/3/5 2/4/5\n"
                        "f 1/1/6 5/2/6 8/3/6\nf 1/1/6 8/3/6 4/4/6\n"
                    , f);
                    fclose(f);
                }
            }
            torchModel = LoadModel("models/torch.obj");
            torchModelLoaded = (torchModel.meshCount > 0);
            if(torchModelLoaded){
                // Apply world shader + a simple brown colour texture
                Image ci = GenImageColor(4,4,{110,72,28,255});
                Texture2D ct = LoadTextureFromImage(ci); UnloadImage(ci);
                torchModel.materials[0].shader = worldShader;
                SetMaterialTexture(&torchModel.materials[0], MATERIAL_MAP_DIFFUSE, ct);
            }
        }


        float fogStart=(RENDER_DISTANCE-1.5f)*CHUNK_SIZE;
        float fogEnd  =(RENDER_DISTANCE-0.3f)*CHUNK_SIZE;
        SetShaderValue(worldShader,locFogStart,&fogStart,SHADER_UNIFORM_FLOAT);
        SetShaderValue(worldShader,locFogEnd,  &fogEnd,  SHADER_UNIFORM_FLOAT);

        skyShader  =LoadShaderFromMemory(SKY_VS,SKY_FS);
        locSkyDun  =GetShaderLocation(skyShader,"sunDir");
        locSunElevLoc=GetShaderLocation(skyShader,"uSunElev");

        Mesh sm=GenMeshSphere(480.0f,16,16);
        skyModel=LoadModelFromMesh(sm);
        skyModel.materials[0].shader=skyShader;

        godRayShader=LoadShaderFromMemory(nullptr,GODRAY_FS);
        locSunUV     =GetShaderLocation(godRayShader,"sunUV");
        locSunVisible=GetShaderLocation(godRayShader,"sunVisible");

        // Launch worker threads for async chunk generation
        // Use half of hardware threads (min 1, max 4) — leave cores for rendering
        bfsQ.reserve(MAX_CHUNK_SLOTS*NUM_SECTIONS);
        bfsQ2.reserve(MAX_CHUNK_SLOTS*NUM_SECTIONS);
        memset(occlusionVis,0,sizeof(occlusionVis));
        memset(slotToKey,-1,sizeof(slotToKey));
        memset(slotToChunk,0,sizeof(slotToChunk));
        // ── Two separate thread pools: gen workers + mesh workers ────────────
        // Gen jobs are ~10-15ms each (full 512-tall 3D noise per chunk).
        // Mesh jobs are ~0.5-2ms each (vertex building for one 16-block section).
        // Mixing them in one pool means gen jobs block mesh jobs → visible load stutter.
        // Dedicated pools mean mesh workers are NEVER blocked waiting for gen.
        //
        // On the Ryzen 7 5700G (8c/16t): 2 gen + 5 mesh workers, 1 core reserved for main thread.
        int hwc = (int)std::thread::hardware_concurrency();
        int nGen  = 2;                                // gen is IO-bound, 2 is plenty
        int nMesh = std::max(2, std::min(5, hwc - 3));// mesh is CPU-bound but short
        for(int i=0;i<nGen;  i++) workers.emplace_back([this]{ GenWorkerFunc();  });
        for(int i=0;i<nMesh; i++) workers.emplace_back([this]{ MeshWorkerFunc(); });
    }

    // Advance day/night cycle and push all lighting uniforms
    void UpdateDepth(float playerY, float playerX, float playerZ){
        // Use actual surface Y at player's XZ position
        int surfY = SurfaceY((int)playerX, (int)playerZ);
        float depth = (float)surfY - playerY; // positive = underground
        // Fade starts 3 blocks below surface top, fully dark 18 blocks below
        depthFade = Clamp((depth - 3.0f) / 18.0f, 0.0f, 1.0f);
    }

    void UpdateTime(float dt){
        const float DAY_DURATION = 8.0f * 60.0f; // 8-minute full day
        dayTime += dt / DAY_DURATION;
        if(dayTime > 1.0f) dayTime -= 1.0f;

        // Sun arcs over the top: angle 0 = east horizon, PI/2 = noon, PI = west
        float angle = dayTime * 2.0f * PI;
        sunDir = Vector3Normalize({cosf(angle), sinf(angle), 0.25f});

        // How far above horizon: smooth to avoid hard cutoff at sunrise/set
        float elev     = sunDir.y; // -1..1
        // daylight: sky palette blend, fades quickly past horizon
        float daylight = Clamp(elev * 3.0f, 0.0f, 1.0f);
        // sunInt: stays near-full AT the horizon (golden hour), fades only once
        // sun drops clearly below (-0.25).  Old formula gave 0.17 at elev=0 (too dark).
        // New: 0.0 at elev=-0.25, ~0.86 at elev=0, 1.0 at elev>=0.15
        float sunInt   = Clamp((elev + 0.25f) / 0.40f, 0.0f, 1.0f);

        // Sun color: orange-white at horizon, near-white at noon.
        // Slightly pulled back from pure 1.0 so top-faces don't overexpose.
        float noon  = Clamp(elev * 2.0f, 0.0f, 1.0f);
        float sunCol[3] = {0.92f, 0.50f + noon*0.38f, 0.16f + noon*0.58f};

        // Sky ambient: blue during day, near-zero at night
        // Sky ambient: always has a cool blue floor even at night
        float skyCol[3] = {0.06f+0.14f*daylight, 0.08f+0.18f*daylight, 0.18f+0.22f*daylight};

        // Fog: warm orange during day, deep blue-black at night
        // Fog: dark night → orange golden hour → pale blue midday
        float tGolden = fmaxf(0.0f, 1.0f - fabsf(elev) / 0.25f) * (elev > -0.15f ? 1.0f : 0.0f);
        tGolden = Clamp(tGolden, 0.0f, 1.0f);
        float tDay2   = Clamp((elev - 0.10f) / 0.35f, 0.0f, 1.0f);
        float fogCol[3] = {
            0.01f + 0.65f*tGolden + 0.60f*tDay2*(1.0f-tGolden),
            0.01f + 0.28f*tGolden + 0.72f*tDay2*(1.0f-tGolden),
            0.04f + 0.05f*tGolden + 0.82f*tDay2*(1.0f-tGolden)
        };

        // Underground: override fog to pitch black cave darkness
        float caveFog[3] = {0.0f, 0.0f, 0.0f};
        fogCol[0] = fogCol[0]*(1.0f-depthFade) + caveFog[0]*depthFade;
        fogCol[1] = fogCol[1]*(1.0f-depthFade) + caveFog[1]*depthFade;
        fogCol[2] = fogCol[2]*(1.0f-depthFade) + caveFog[2]*depthFade;
        // Also tighten fog range underground so caves feel dark quickly
        float fogStart = (RENDER_DISTANCE-1.5f)*CHUNK_SIZE * (1.0f - depthFade*0.6f);
        float fogEnd   = (RENDER_DISTANCE-0.3f)*CHUNK_SIZE * (1.0f - depthFade*0.5f);
        SetShaderValue(worldShader, locFogStart, &fogStart, SHADER_UNIFORM_FLOAT);
        SetShaderValue(worldShader, locFogEnd,   &fogEnd,   SHADER_UNIFORM_FLOAT);

        // Push to shaders
        float sdv[3]={sunDir.x,sunDir.y,sunDir.z};
        SetShaderValue(skyShader,   locSkyDun,      sdv,     SHADER_UNIFORM_VEC3);
        SetShaderValue(skyShader,   locSunElevLoc,  &elev,    SHADER_UNIFORM_FLOAT);
        SetShaderValue(worldShader, locSunColor,    sunCol,  SHADER_UNIFORM_VEC3);
        SetShaderValue(worldShader, locSunIntensity,&sunInt, SHADER_UNIFORM_FLOAT);
        SetShaderValue(worldShader, locSkyColor,    skyCol,  SHADER_UNIFORM_VEC3);
        SetShaderValue(worldShader, locFogColor,    fogCol,  SHADER_UNIFORM_VEC3);
        SetShaderValue(worldShader, locSunDir,     sdv,    SHADER_UNIFORM_VEC3);
    }

    void UpdateTorchLights(Vector3 camPos){
        // ── Rebuild cache only when world changes (block placed/removed, chunk load/unload)
        if(torchCacheDirty){
            torchCache.clear();
            for(auto& [k,c]:chunks){
                float cWX=(float)(c.chunkX*CHUNK_SIZE), cWZ=(float)(c.chunkZ*CHUNK_SIZE);
                // Torches can only exist at or below the surface — skip air above terrain.
                // Also skip sections flagged all-solid (no air, no torches possible).
                for(int x=0;x<CHUNK_SIZE;x++) for(int z=0;z<CHUNK_SIZE;z++){
                    int maxY = c.surfaceH[x][z]; // no blocks above this
                    for(int y=0;y<=maxY;y++){
                        // Skip solid sections early via section stride
                        int sec=y/SECTION_HEIGHT;
                        if(c.secAllSolid[sec]){ y=(sec+1)*SECTION_HEIGHT-1; continue; }
                        if(c.Get(x,y,z)==BLOCK_TORCH)
                            torchCache.push_back({cWX+x+0.5f,(float)y+0.9f,cWZ+z+0.5f});
                    }
                }
            }
            torchCacheDirty=false;
        }

        // ── Every frame: just filter the tiny cache by distance ──────────
        struct TL { float dist2; float x,y,z; };
        std::vector<TL> lights;
        visibleTorches.clear();
        for(auto& t : torchCache){
            float dx=t.x-camPos.x, dy=t.y-camPos.y, dz=t.z-camPos.z;
            float d2=dx*dx+dy*dy+dz*dz;
            if(d2 < 24.0f*24.0f) visibleTorches.push_back(t);
            if(d2 < 20.0f*20.0f) lights.push_back({d2,t.x,t.y,t.z});
        }

        std::sort(lights.begin(),lights.end(),[](const TL& a,const TL& b){ return a.dist2<b.dist2; });
        if((int)lights.size()>8) lights.resize(8); // 8 lights max — halves per-fragment loop vs 16

        int n=(int)lights.size();
        SetShaderValue(worldShader, locLightCount, &n, SHADER_UNIFORM_INT);
        if(n>0){
            float posArr[8*3]={}, colArr[8*3]={};
            for(int i=0;i<n;i++){
                posArr[i*3+0]=lights[i].x; posArr[i*3+1]=lights[i].y; posArr[i*3+2]=lights[i].z;
                colArr[i*3+0]=1.4f;        colArr[i*3+1]=0.7f;        colArr[i*3+2]=0.2f;
            }
            SetShaderValueV(worldShader, locLightPos,   posArr, SHADER_UNIFORM_VEC3, n);
            SetShaderValueV(worldShader, locLightColor, colArr, SHADER_UNIFORM_VEC3, n);
        }
    }

    // Draw all visible torches as 3D geometry: stick + flickering billboard flame
    void DrawTorches(Camera3D& cam, float time){
        // ── Stick: one DrawModelEx per torch (pre-uploaded VBO, fast) ──────
        // The OBJ is in block-local space (0..1). We offset so the block
        // centre aligns with the torch grid position stored in visibleTorches.
        if(torchModelLoaded){
            for(auto& t : visibleTorches){
                // t.x/z are already block-centre; t.y is y+0.9 (flame pos).
                // OBJ origin is at block corner, so shift by -0.5 on X/Z and
                // back to the block floor on Y.
                Vector3 origin = {t.x - 0.5f, t.y - 0.9f, t.z - 0.5f};
                DrawModelEx(torchModel, origin, {0,1,0}, 0.0f, {1,1,1}, WHITE);
            }
        }

        // ── Flame: DrawBillboard — single textured quad, always faces camera ─
        // Much faster than rlBegin and looks great for fire.
        Vector3 up = {0,1,0};
        for(auto& t : visibleTorches){
            float flicker = 1.0f + sinf(time*9.0f + t.x*3.7f + t.z*2.1f)*0.09f;
            float fsize = 0.38f * flicker;
            // Flame centre: top of stick (y+0.65) + half flame height
            float fy = t.y - 0.9f + 0.65f + fsize * 0.4f;
            DrawBillboard(cam, flameTex, {t.x, fy, t.z}, fsize, {255,200,80,210});
        }
    }

    void UpdateCamPos(Vector3 pos){
        float cp[3]={pos.x,pos.y,pos.z};
        SetShaderValue(worldShader,locCamPos,cp,SHADER_UNIFORM_VEC3);
    }

    int64_t Key(int cx,int cz){ return ((int64_t)(uint32_t)cx<<32)|(uint32_t)cz; }
    Chunk* Get(int cx,int cz){auto it=chunks.find(Key(cx,cz));return it!=chunks.end()?&it->second:nullptr;}
    Chunk* Get_byKey(int64_t k){auto it=chunks.find(k);return it!=chunks.end()?&it->second:nullptr;}

    BlockType GetBlock(int wx,int wy,int wz){
        if(wy<0||wy>=CHUNK_HEIGHT) return BLOCK_AIR;
        int cx=(int)floor((float)wx/CHUNK_SIZE),cz=(int)floor((float)wz/CHUNK_SIZE);
        Chunk* c=Get(cx,cz); if(!c) return BLOCK_AIR;
        return c->Get(wx-cx*CHUNK_SIZE, wy, wz-cz*CHUNK_SIZE);
    }
    bool IsSolid(int wx,int wy,int wz){
        BlockType t=GetBlock(wx,wy,wz);
        return t!=BLOCK_AIR && t!=BLOCK_TORCH;
    }

    void SetBlock(int wx,int wy,int wz,BlockType t){
        if(wy<0||wy>=CHUNK_HEIGHT) return;
        int cx=(int)floor((float)wx/CHUNK_SIZE),cz=(int)floor((float)wz/CHUNK_SIZE);
        Chunk* c=Get(cx,cz); if(!c) return;
        int lx=wx-cx*CHUNK_SIZE, lz=wz-cz*CHUNK_SIZE;
        BlockType old = c->Get(lx,wy,lz);
        c->Set(lx,wy,lz,t);
        // If a torch was placed or removed, invalidate the cache
        if(t==BLOCK_TORCH || old==BLOCK_TORCH) torchCacheDirty=true;
        int idx=lx*CHUNK_HEIGHT*CHUNK_SIZE + wy*CHUNK_SIZE + lz;
        mods[Key(cx,cz)][idx]=t;

        // Dirty every section that could have a newly exposed or newly hidden face.
        // A block change at (lx,wy,lz) can affect faces in its own section AND in
        // any adjacent section — specifically when the block sits on a section boundary
        // (e.g. breaking Y=16 exposes the top face of Y=15, which lives in the section below).
        // We also dirty neighbor-chunk sections for blocks on chunk X/Z borders.
        int sec = wy / SECTION_HEIGHT;

        // Own section is always dirty (c->Set already set secDirty[sec])

        // Section above (block's +Y face or the face below is in sec+1)
        if(wy % SECTION_HEIGHT == SECTION_HEIGHT-1 && sec+1 < NUM_SECTIONS)
            c->secDirty[sec+1] = true;
        // Section below (block's -Y face or the face above is in sec-1)
        if(wy % SECTION_HEIGHT == 0 && sec-1 >= 0)
            c->secDirty[sec-1] = true;

        // Chunk-border X neighbors — same section, and section above/below if at boundary
        if(lx==0){
            if(Chunk* n=Get(cx-1,cz)){
                n->secDirty[sec]=true;
                if(wy%SECTION_HEIGHT==SECTION_HEIGHT-1 && sec+1<NUM_SECTIONS) n->secDirty[sec+1]=true;
                if(wy%SECTION_HEIGHT==0              && sec-1>=0)             n->secDirty[sec-1]=true;
            }
        }
        if(lx==CHUNK_SIZE-1){
            if(Chunk* n=Get(cx+1,cz)){
                n->secDirty[sec]=true;
                if(wy%SECTION_HEIGHT==SECTION_HEIGHT-1 && sec+1<NUM_SECTIONS) n->secDirty[sec+1]=true;
                if(wy%SECTION_HEIGHT==0              && sec-1>=0)             n->secDirty[sec-1]=true;
            }
        }
        // Chunk-border Z neighbors
        if(lz==0){
            if(Chunk* n=Get(cx,cz-1)){
                n->secDirty[sec]=true;
                if(wy%SECTION_HEIGHT==SECTION_HEIGHT-1 && sec+1<NUM_SECTIONS) n->secDirty[sec+1]=true;
                if(wy%SECTION_HEIGHT==0              && sec-1>=0)             n->secDirty[sec-1]=true;
            }
        }
        if(lz==CHUNK_SIZE-1){
            if(Chunk* n=Get(cx,cz+1)){
                n->secDirty[sec]=true;
                if(wy%SECTION_HEIGHT==SECTION_HEIGHT-1 && sec+1<NUM_SECTIONS) n->secDirty[sec+1]=true;
                if(wy%SECTION_HEIGHT==0              && sec-1>=0)             n->secDirty[sec-1]=true;
            }
        }
    }

    // ── Gen worker: purely handles world generation, never touches meshes ─────
    // Each gen job is ~10-15ms (3D noise for 512-tall chunk). Keeping gen workers
    // separate means mesh workers are NEVER blocked waiting for a gen job to finish.
    void GenWorkerFunc(){
        FastNoiseLite wNoise, wCave, wOre;
        wNoise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        wNoise.SetFrequency(0.07f);  wNoise.SetSeed(1337);
        wCave.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        wCave.SetFrequency(0.025f);  wCave.SetSeed(9182);
        wOre.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        wOre.SetFrequency(0.18f);    wOre.SetSeed(4455);

        while(true){
            GenJob job;
            {
                std::unique_lock<std::mutex> lk(pendingMutex);
                pendingCV.wait(lk,[this]{
                    return stopWorkers.load() || !pendingJobs.empty();
                });
                if(stopWorkers && pendingJobs.empty()) return;
                job=std::move(pendingJobs.front());
                pendingJobs.pop();
            }
            // Generate block data entirely off the main thread
            ReadyChunk rc;
            rc.cx=job.cx; rc.cz=job.cz;
            rc.blocks.assign(CHUNK_SIZE*CHUNK_HEIGHT*CHUNK_SIZE, BLOCK_AIR);
            memset(rc.surfH, 0, sizeof(rc.surfH));

            for(int x=0;x<CHUNK_SIZE;x++) for(int z=0;z<CHUNK_SIZE;z++){
                int wx=job.cx*CHUNK_SIZE+x, wz=job.cz*CHUNK_SIZE+z;
                float n=wNoise.GetNoise((float)wx,(float)wz);
                int h=(int)(n*12.0f)+440;
                for(int y=0;y<CHUNK_HEIGHT;y++){
                    int idx=x*CHUNK_HEIGHT*CHUNK_SIZE+y*CHUNK_SIZE+z;
                    BlockType base;
                    if(y<2){ rc.blocks[idx]=BLOCK_BEDROCK; continue; }
                    if(y>h)          base=BLOCK_AIR;
                    else if(y==h)    base=BLOCK_GRASS;
                    else if(y>=h-3)  base=BLOCK_DIRT;
                    else              base=BLOCK_STONE;
                    if(base!=BLOCK_AIR && y<h-2){
                        float cv1=wCave.GetNoise((float)wx,(float)y,(float)wz);
                        float cv2=wCave.GetNoise((float)wx+37.5f,(float)y*1.3f,(float)wz+19.1f);
                        float dT=1.0f-(float)y/(float)(h-2); dT*=dT;
                        if(cv1*cv1+cv2*cv2 < 0.006f+dT*0.025f) base=BLOCK_AIR;
                    }
                    if(base==BLOCK_STONE){
                        float ov =wOre.GetNoise((float)wx,(float)y,(float)wz);
                        float ov2=wOre.GetNoise((float)wx*1.7f+100,(float)y*1.7f,(float)wz*1.7f+100);
                        if     (y<=380&&ov>0.55f&&ov2>0.10f) base=BLOCK_COAL;
                        else if(y<=300&&ov>0.62f&&ov2>0.15f) base=BLOCK_IRON;
                        else if(y<=180&&ov>0.70f&&ov2>0.20f) base=BLOCK_GOLD;
                        else if(y<=80 &&ov>0.76f&&ov2>0.25f) base=BLOCK_DIAMOND;
                    }
                    rc.blocks[idx]=base;
                }
                for(auto& [midx,mt]:job.savedMods){
                    int mx=midx/(CHUNK_HEIGHT*CHUNK_SIZE);
                    int my=(midx/CHUNK_SIZE)%CHUNK_HEIGHT;
                    int mz=midx%CHUNK_SIZE;
                    rc.blocks[mx*CHUNK_HEIGHT*CHUNK_SIZE+my*CHUNK_SIZE+mz]=mt;
                }
                rc.surfH[x][z]=h;
            }
            for(int x=0;x<CHUNK_SIZE;x++) for(int z=0;z<CHUNK_SIZE;z++){
                for(int y=CHUNK_HEIGHT-1;y>=0;y--)
                    if(rc.blocks[x*CHUNK_HEIGHT*CHUNK_SIZE+y*CHUNK_SIZE+z]!=BLOCK_AIR){
                        rc.surfH[x][z]=y; break;
                    }
            }
            for(int s=0;s<NUM_SECTIONS;s++){
                int yMin=s*SECTION_HEIGHT, yMax=yMin+SECTION_HEIGHT;
                bool allSolid=true;
                for(int x=0;x<CHUNK_SIZE&&allSolid;x++)
                for(int y=yMin;y<yMax&&allSolid;y++)
                for(int z=0;z<CHUNK_SIZE&&allSolid;z++)
                    if(rc.blocks[x*CHUNK_HEIGHT*CHUNK_SIZE+y*CHUNK_SIZE+z]==BLOCK_AIR)
                        allSolid=false;
                rc.secAllSolid[s]=allSolid;
                ComputeSectionFaceGraph(rc.blocks.data(), CHUNK_HEIGHT, CHUNK_SIZE,
                                        yMin, rc.secFaceGraph[s]);
            }
            { std::lock_guard<std::mutex> lk(readyMutex); readyChunks.push(std::move(rc)); }
        }
    }

    // ── Mesh worker: purely builds vertex data for sections ───────────────────
    // These are fast (~0.5-2ms) and latency-sensitive — having dedicated workers
    // means they're never blocked by a slow gen job and chunks appear much faster.
    void MeshWorkerFunc(){
        while(true){
            std::unique_ptr<MeshJob> mj;
            {
                std::unique_lock<std::mutex> lk(meshJobMutex);
                meshJobCV.wait(lk,[this]{
                    return stopWorkers.load() || !meshJobs.empty();
                });
                if(stopWorkers && meshJobs.empty()) return;
                if(meshJobs.empty()) continue;
                mj=std::move(const_cast<std::unique_ptr<MeshJob>&>(meshJobs.top()));
                meshJobs.pop();
            }
            ReadyMesh rm;
            rm.cx=mj->cx; rm.cz=mj->cz; rm.sec=mj->sec;
            FillMeshData(*mj,rm);
            { std::lock_guard<std::mutex> lk(meshReadyMutex); meshReady.push(std::move(rm)); }
        }
    }

    // Worker thread: pops generation jobs, runs noise, posts ReadyChunk
    // LEGACY — kept only so the compiler finds it; Init() now calls Gen/MeshWorkerFunc directly
    void WorkerFunc(){
        // Each worker gets its own noise instances (FastNoiseLite is not thread-safe)
        FastNoiseLite wNoise, wCave, wOre;
        wNoise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        wNoise.SetFrequency(0.07f);  wNoise.SetSeed(1337);
        wCave.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        wCave.SetFrequency(0.025f);  wCave.SetSeed(9182);
        wOre.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
        wOre.SetFrequency(0.18f);    wOre.SetSeed(4455);

        while(true){
            // Check mesh jobs first (higher priority — latency-sensitive)
            {
                std::unique_ptr<MeshJob> mj;
                { std::lock_guard<std::mutex> lk(meshJobMutex);
                  if(!meshJobs.empty()){
                      // priority_queue: const top() — must const_cast to move out
                      mj=std::move(const_cast<std::unique_ptr<MeshJob>&>(meshJobs.top()));
                      meshJobs.pop();
                  } }
                if(mj){
                    ReadyMesh rm;
                    rm.cx=mj->cx; rm.cz=mj->cz; rm.sec=mj->sec;
                    FillMeshData(*mj,rm); // allocates MemAlloc buffers in rm
                    { std::lock_guard<std::mutex> lk(meshReadyMutex); meshReady.push(std::move(rm)); }
                    continue;
                }
            }
            GenJob job;
            {
                std::unique_lock<std::mutex> lk(pendingMutex);
                pendingCV.wait(lk,[this]{
                    if(stopWorkers.load()) return true;
                    if(!pendingJobs.empty()) return true;
                    std::lock_guard<std::mutex> l(meshJobMutex);
                    return !meshJobs.empty();  // unique_ptr queue — same .empty() API
                });
                if(stopWorkers && pendingJobs.empty()) return;
                // Woken for a mesh job, not a gen job — loop back to handle it
                if(pendingJobs.empty()) continue;
                job=std::move(pendingJobs.front());
                pendingJobs.pop();
            }
            // Generate block data entirely off the main thread
            ReadyChunk rc;
            rc.cx=job.cx; rc.cz=job.cz;
            rc.blocks.assign(CHUNK_SIZE*CHUNK_HEIGHT*CHUNK_SIZE, BLOCK_AIR);
            memset(rc.surfH, 0, sizeof(rc.surfH));

            for(int x=0;x<CHUNK_SIZE;x++) for(int z=0;z<CHUNK_SIZE;z++){
                int wx=job.cx*CHUNK_SIZE+x, wz=job.cz*CHUNK_SIZE+z;
                float n=wNoise.GetNoise((float)wx,(float)wz);
                int h=(int)(n*12.0f)+440;
                for(int y=0;y<CHUNK_HEIGHT;y++){
                    int idx=x*CHUNK_HEIGHT*CHUNK_SIZE+y*CHUNK_SIZE+z;
                    BlockType base;
                    if(y<2){ rc.blocks[idx]=BLOCK_BEDROCK; continue; }
                    if(y>h)          base=BLOCK_AIR;
                    else if(y==h)    base=BLOCK_GRASS;
                    else if(y>=h-3)  base=BLOCK_DIRT;
                    else              base=BLOCK_STONE;
                    if(base!=BLOCK_AIR && y<h-2){
                        float cv1=wCave.GetNoise((float)wx,(float)y,(float)wz);
                        float cv2=wCave.GetNoise((float)wx+37.5f,(float)y*1.3f,(float)wz+19.1f);
                        float dT=1.0f-(float)y/(float)(h-2); dT*=dT;
                        if(cv1*cv1+cv2*cv2 < 0.006f+dT*0.025f) base=BLOCK_AIR;
                    }
                    if(base==BLOCK_STONE){
                        float ov =wOre.GetNoise((float)wx,(float)y,(float)wz);
                        float ov2=wOre.GetNoise((float)wx*1.7f+100,(float)y*1.7f,(float)wz*1.7f+100);
                        if     (y<=380&&ov>0.55f&&ov2>0.10f) base=BLOCK_COAL;
                        else if(y<=300&&ov>0.62f&&ov2>0.15f) base=BLOCK_IRON;
                        else if(y<=180&&ov>0.70f&&ov2>0.20f) base=BLOCK_GOLD;
                        else if(y<=80 &&ov>0.76f&&ov2>0.25f) base=BLOCK_DIAMOND;
                    }
                    rc.blocks[idx]=base;
                }
                // Apply saved mods for this column
                for(auto& [midx,mt]:job.savedMods){
                    int mx=midx/(CHUNK_HEIGHT*CHUNK_SIZE);
                    int my=(midx/CHUNK_SIZE)%CHUNK_HEIGHT;
                    int mz=midx%CHUNK_SIZE;
                    rc.blocks[mx*CHUNK_HEIGHT*CHUNK_SIZE+my*CHUNK_SIZE+mz]=mt;
                }
                // Bake surfaceH for this column
                rc.surfH[x][z]=h;
            }
            // Final surfH pass from actual blocks (handles mods)
            for(int x=0;x<CHUNK_SIZE;x++) for(int z=0;z<CHUNK_SIZE;z++){
                for(int y=CHUNK_HEIGHT-1;y>=0;y--)
                    if(rc.blocks[x*CHUNK_HEIGHT*CHUNK_SIZE+y*CHUNK_SIZE+z]!=BLOCK_AIR){
                        rc.surfH[x][z]=y; break;
                    }
            }
            // Compute secAllSolid + face connectivity graph for occlusion culling
            for(int s=0;s<NUM_SECTIONS;s++){
                int yMin=s*SECTION_HEIGHT, yMax=yMin+SECTION_HEIGHT;
                bool allSolid=true;
                for(int x=0;x<CHUNK_SIZE&&allSolid;x++)
                for(int y=yMin;y<yMax&&allSolid;y++)
                for(int z=0;z<CHUNK_SIZE&&allSolid;z++)
                    if(rc.blocks[x*CHUNK_HEIGHT*CHUNK_SIZE+y*CHUNK_SIZE+z]==BLOCK_AIR)
                        allSolid=false;
                rc.secAllSolid[s]=allSolid;
                ComputeSectionFaceGraph(rc.blocks.data(), CHUNK_HEIGHT, CHUNK_SIZE,
                                        yMin, rc.secFaceGraph[s]);
            }
            { std::lock_guard<std::mutex> lk(readyMutex); readyChunks.push(std::move(rc)); }
        }
    }

    // Generate all chunks in radius with NO throttle — used at startup only
    void GenerateImmediate(int cx, int cz, int radius){
        for(int x=cx-radius;x<=cx+radius;x++)
        for(int z=cz-radius;z<=cz+radius;z++){
            int64_t k=Key(x,z);
            if(chunks.find(k)==chunks.end()){
                auto& c=chunks[k];
                c.chunkX=x; c.chunkZ=z;
                if(nextFreeSlot < MAX_CHUNK_SLOTS){
                    c.occSlot = nextFreeSlot++;
                    slotToKey[c.occSlot]   = k;
                    slotToChunk[c.occSlot] = &c;
                }
                auto mit=mods.find(k);
                c.Generate(noise, caveNoise, oreNoise, mit!=mods.end()?&mit->second:nullptr);
            }
        }
        // Dirty neighbours after all chunks exist so seams are correct
        for(int x=cx-radius;x<=cx+radius;x++)
        for(int z=cz-radius;z<=cz+radius;z++){
            for(auto* nb:{Get(x-1,z),Get(x+1,z),Get(x,z-1),Get(x,z+1)}) if(nb) nb->MarkBorderDirty();
        }
    }

    // Find the highest solid block at world (wx, wz), starting from top
    int SurfaceY(int wx, int wz){
        int cx=(int)floor((float)wx/CHUNK_SIZE), cz=(int)floor((float)wz/CHUNK_SIZE);
        Chunk* c=Get(cx,cz); if(!c) return 452;
        int lx=wx-cx*CHUNK_SIZE, lz=wz-cz*CHUNK_SIZE;
        // Use precomputed surface height — O(1) instead of scanning 512 blocks every frame
        return c->surfaceH[lx][lz];
    }

    void Update(Vector3 camPos){
        int cx=(int)floor(camPos.x/CHUNK_SIZE),cz=(int)floor(camPos.z/CHUNK_SIZE);
        lastCX=cx; lastCZ=cz;

        // ── Drain completed async jobs (main thread only — safe to touch GPU) ──
        {
            std::lock_guard<std::mutex> lk(readyMutex);
            int inserted=0;
            while(!readyChunks.empty() && inserted<4){
                ReadyChunk rc=std::move(readyChunks.front());
                readyChunks.pop();
                int64_t k=Key(rc.cx,rc.cz);
                inFlight.erase(k);
                if(chunks.find(k)==chunks.end()){
                    auto& c=chunks[k];
                    c.chunkX=rc.cx; c.chunkZ=rc.cz;
                    c.blocks=std::move(rc.blocks);
                    memcpy(c.surfaceH,    rc.surfH,       sizeof(c.surfaceH));
                    memcpy(c.secAllSolid,   rc.secAllSolid,   sizeof(c.secAllSolid));
                    memcpy(c.secFaceGraph, rc.secFaceGraph, sizeof(c.secFaceGraph));
                    c.MarkAllDirty();
                    torchCacheDirty=true;
                    sortedDirty=true;
                    for(auto* nb:{Get(rc.cx-1,rc.cz),Get(rc.cx+1,rc.cz),
                                   Get(rc.cx,rc.cz-1),Get(rc.cx,rc.cz+1)})
                        if(nb) nb->MarkBorderDirty(); // only non-solid sections need seam update
                }
                inserted++;
            }
        }

        // ── Queue missing chunks for async generation ─────────────────────────
        for(int x=cx-RENDER_DISTANCE;x<=cx+RENDER_DISTANCE;x++)
        for(int z=cz-RENDER_DISTANCE;z<=cz+RENDER_DISTANCE;z++){
            int64_t k=Key(x,z);
            if(chunks.find(k)==chunks.end() && inFlight.find(k)==inFlight.end()){
                inFlight.insert(k);
                GenJob job;
                job.cx=x; job.cz=z;
                auto mit=mods.find(k);
                if(mit!=mods.end()) job.savedMods=mit->second;
                { std::lock_guard<std::mutex> lk(pendingMutex); pendingJobs.push(std::move(job)); }
                pendingCV.notify_one();
            }
        }

        // ── Unload distant chunks ─────────────────────────────────────────────
        std::vector<int64_t> rem;
        for(auto& [k,c]:chunks)
            if(abs(c.chunkX-cx)>RENDER_DISTANCE+1||abs(c.chunkZ-cz)>RENDER_DISTANCE+1) rem.push_back(k);
        if(!rem.empty()){ torchCacheDirty=true; sortedDirty=true; }
        for(auto& k:rem){
            auto& c=chunks[k];
            if(c.meshJobsInFlight>0) continue;
            // Before erasing, tell all 4 neighbours to rebuild their border sections.
            // Without this, the neighbours had their border faces omitted (the face was
            // between them and this chunk, which was solid and thus culled). Once this
            // chunk disappears those faces are exposed — but the neighbours won't know
            // to rebuild unless we dirty them here.
            for(auto* nb:{Get(c.chunkX-1,c.chunkZ),Get(c.chunkX+1,c.chunkZ),
                           Get(c.chunkX,c.chunkZ-1),Get(c.chunkX,c.chunkZ+1)})
                if(nb) nb->MarkAllDirty();
            c.Unload(); chunks.erase(k);
        }
    }

    void DrawSky(Vector3 eye){
        // Skip sky the moment the player starts going underground.
        // Even a tiny depthFade means the player is below the surface — drawing the
        // sky dome then causes it to bleed through cave ceilings when looking up.
        if(depthFade > 0.0f) return;
        rlDisableBackfaceCulling(); rlDisableDepthMask();
        DrawModel(skyModel, eye, 1.0f, WHITE);
        rlEnableDepthMask(); rlEnableBackfaceCulling();
    }

    // Build a MeshJob snapshot from live chunk data and push to worker queue.
    // Called on main thread — reads live chunk data safely (no workers write chunks).
    void SubmitMeshJob(Chunk& c, int sec, Chunk* cnx, Chunk* cpx, Chunk* cnz, Chunk* cpz, Vector3 cam){
        MeshJob job;
        job.cx=c.chunkX; job.cz=c.chunkZ; job.sec=sec;
        job.yMin=sec*SECTION_HEIGHT;
        float sx=(c.chunkX+0.5f)*CHUNK_SIZE, sy=(sec+0.5f)*SECTION_HEIGHT, sz=(c.chunkZ+0.5f)*CHUNK_SIZE;
        float dx=sx-cam.x, dy=sy-cam.y, dz=sz-cam.z;
        job.dist2=dx*dx+dy*dy*0.25f+dz*dz;
        int yLo=job.yMin-1, yHi=job.yMin+SECTION_HEIGHT;

        // Live pointer — no copy of own section data (18KB → 8 bytes)
        // Chunk is pinned (meshJobsInFlight++) so it won't be erased while worker reads
        job.liveBlocks = c.blocks.data();

        // surfH snapshot — cheap (1KB) and needed for sky factor
        memcpy(job.surfH, c.surfaceH, sizeof(job.surfH));

        // Neighbour borders — must snapshot (neighbours can be unloaded independently)
        for(int z=0;z<CHUNK_SIZE;z++) for(int y=yLo;y<=yHi;y++){
            int yo=y-yLo;
            job.nbNX[yo][z]=cnx?cnx->Get(CHUNK_SIZE-1,y,z):BLOCK_AIR;
            job.nbPX[yo][z]=cpx?cpx->Get(0,y,z):BLOCK_AIR;
        }
        for(int x=0;x<CHUNK_SIZE;x++) for(int y=yLo;y<=yHi;y++){
            int yo=y-yLo;
            job.nbNZ[x][yo]=cnz?cnz->Get(x,y,CHUNK_SIZE-1):BLOCK_AIR;
            job.nbPZ[x][yo]=cpz?cpz->Get(x,y,0):BLOCK_AIR;
        }
        for(int i=0;i<CHUNK_SIZE;i++){
            job.snbNX[i]=cnx?cnx->surfaceH[CHUNK_SIZE-1][i]:CHUNK_HEIGHT;
            job.snbPX[i]=cpx?cpx->surfaceH[0][i]:CHUNK_HEIGHT;
            job.snbNZ[i]=cnz?cnz->surfaceH[i][CHUNK_SIZE-1]:CHUNK_HEIGHT;
            job.snbPZ[i]=cpz?cpz->surfaceH[i][0]:CHUNK_HEIGHT;
        }
        c.meshJobsInFlight++;
        c.secMeshInFlight[sec]=true;
        { std::lock_guard<std::mutex> lk(meshJobMutex); meshJobs.push(std::make_unique<MeshJob>(std::move(job))); }
        meshJobCV.notify_all(); // wake mesh workers only — gen workers don't handle mesh jobs
    }

    // Queue a mesh for deletion next frame — avoids glDeleteBuffers stalling the GPU
    // while it's still rendering the current frame with those buffers.
    void DeferUnloadMesh(Mesh& m){
        if(m.vertexCount>0){ meshDeleteQueue.push_back(m); m={}; }
    }

    void DrawTerrain(float playerY, Camera3D& cam, int SW, int SH){
        // ── Process deferred mesh deletes (GPU finished previous frame) ───────
        // Delete at most 4 old buffers per frame — spreads driver overhead
        for(int i=0; i<4 && !meshDeleteQueue.empty(); i++){
            UnloadMesh(meshDeleteQueue.back());
            meshDeleteQueue.pop_back();
        }

        // ── Drain completed mesh jobs — hard count cap + time budget ─────────────
        // Caps at MAX_UPLOADS/frame to prevent a burst of 32 sections from one new chunk
        // causing a 6+ ms spike on an integrated GPU. Workers keep filling the queue
        // between frames so visual pop-in is minimal.
        {
            float lastFrame = GetFrameTime() * 1000.0f;
            const double BUDGET_MS = (double)Clamp(lastFrame * 0.30f, 0.3f, 2.5f);
            const int    MAX_UPLOADS = 4;
            double tStart = GetTime() * 1000.0;
            int uploads = 0;

            while(uploads < MAX_UPLOADS){
                if(GetTime()*1000.0 - tStart > BUDGET_MS) break;
                ReadyMesh rm;
                {
                    std::lock_guard<std::mutex> lk(meshReadyMutex);
                    if(meshReady.empty()) break;
                    rm = std::move(meshReady.front());
                    meshReady.pop();
                }
                Chunk* c = Get(rm.cx, rm.cz);
                if(!c){ rm.Free(); continue; }
                int s = rm.sec;
                DeferUnloadMesh(c->meshes[s]);
                if(rm.vertCount>0){
                    c->meshes[s].vertexCount  = rm.vertCount;
                    c->meshes[s].triangleCount= rm.triCount;
                    c->meshes[s].vertices     = rm.verts;
                    c->meshes[s].texcoords    = rm.uvs;
                    c->meshes[s].colors       = rm.cols;
                    c->meshes[s].indices      = rm.ids;
                    rm.verts=nullptr; rm.uvs=nullptr; rm.cols=nullptr; rm.ids=nullptr;
                    UploadMesh(&c->meshes[s], true); // DYNAMIC_DRAW — no pipeline stall
                }
                c->secMeshInFlight[s] = false;
                c->secDirty[s]        = false;
                c->meshJobsInFlight--;
                uploads++;
            }
        }

        // ── Visibility: pure frustum + vertical range — no BFS ─────────────────
        // The BFS occlusion system was causing holes because:
        //   1. secAllSolid sections had their meshes freed; when a neighbour later
        //      unloaded those faces became exposed but the mesh was gone.
        //   2. BFS through cave sections frequently failed to propagate, blacking out
        //      large parts of the screen underground.
        //   3. At RENDER_DISTANCE=3 there are at most ~50 chunks × 4-6 visible sections
        //      = ~200 draw calls — cheap enough that occlusion culling saves nothing.
        // Simple rule: if the section AABB is in the frustum AND within vertical range,
        // draw it. That's it. Correct by construction.

        // Vertical range: adapts to whether the player is underground.
        //
        // Surface (depthFade=0): ±64 so tall cliffs, overhangs, deep ravines all render.
        // Underground (depthFade=1): upward cull tightens to just ~20 blocks above the
        //   player — enough to see the cave ceiling but not the surface terrain or sky.
        //   Downward stays generous so vertical cave shafts look correct.
        //
        // depthFade is 0 at surface, 1 fully underground (set by UpdateDepth).
        // We interpolate so the transition is smooth as you enter a cave.
        const int CULL_UP   = (int)Clamp(64.0f - depthFade * 44.0f, 20.0f, 64.0f); // 64→20
        const int CULL_DOWN = 64;

        FrustumPlane frustum[6];
        ExtractFrustum(cam, SW, SH, frustum);

        // Front-to-back sort — only rebuilds when chunk set or player chunk changes
        Vector3 camPos = cam.position;
        int curSortCX=(int)floor(camPos.x/CHUNK_SIZE);
        int curSortCZ=(int)floor(camPos.z/CHUNK_SIZE);
        if(sortedDirty || curSortCX!=lastSortCX || curSortCZ!=lastSortCZ){
            sortedChunksCache.clear();
            for(auto& [k,c]:chunks) sortedChunksCache.push_back(&c);
            std::sort(sortedChunksCache.begin(),sortedChunksCache.end(),[&](Chunk* a,Chunk* b){
                float ax=(a->chunkX+0.5f)*CHUNK_SIZE-camPos.x, az=(a->chunkZ+0.5f)*CHUNK_SIZE-camPos.z;
                float bx=(b->chunkX+0.5f)*CHUNK_SIZE-camPos.x, bz=(b->chunkZ+0.5f)*CHUNK_SIZE-camPos.z;
                return ax*ax+az*az < bx*bx+bz*bz;
            });
            sortedDirty=false; lastSortCX=curSortCX; lastSortCZ=curSortCZ;
        }

        int meshSubmitsThisFrame = 0;
        const int MAX_MESH_SUBMITS = 12;

        for(Chunk* cp : sortedChunksCache){ auto& c=*cp;
            Chunk* cnx=Get(c.chunkX-1,c.chunkZ), *cpx=Get(c.chunkX+1,c.chunkZ);
            Chunk* cnz=Get(c.chunkX,c.chunkZ-1), *cpz=Get(c.chunkX,c.chunkZ+1);
            float cWX=(float)(c.chunkX*CHUNK_SIZE), cWZ=(float)(c.chunkZ*CHUNK_SIZE);

            // Whole-chunk horizontal frustum cull (cheap, eliminates ~60% immediately)
            Vector3 cMin={cWX,0,cWZ}, cMax={cWX+CHUNK_SIZE,(float)CHUNK_HEIGHT,cWZ+CHUNK_SIZE};
            if(!AABBInFrustum(frustum,cMin,cMax)) continue;

            for(int s=0;s<NUM_SECTIONS;s++){
                int sYMin=s*SECTION_HEIGHT, sYMax=sYMin+SECTION_HEIGHT;

                // Vertical range: sections far above/below are never visible
                if(sYMax < playerY-CULL_DOWN || sYMin > playerY+CULL_UP){
                    // Free mesh if we have one — saves VRAM on integrated GPU
                    if(c.meshes[s].vertexCount>0){ DeferUnloadMesh(c.meshes[s]); c.secDirty[s]=true; }
                    continue;
                }

                // Per-section frustum test
                Vector3 sMin={cWX,(float)sYMin,cWZ}, sMax={cWX+CHUNK_SIZE,(float)sYMax,cWZ+CHUNK_SIZE};
                bool inFrustum = AABBInFrustum(frustum,sMin,sMax);

                // Queue a mesh build if the section is stale
                if(c.secDirty[s] && !c.secMeshInFlight[s] && meshSubmitsThisFrame < MAX_MESH_SUBMITS){
                    // Prioritise in-frustum sections; allow 1 out-of-frustum build/frame
                    // so background sections are ready before the player turns.
                    if(inFrustum || meshSubmitsThisFrame == 0){
                        SubmitMeshJob(c,s,cnx,cpx,cnz,cpz,camPos);
                        meshSubmitsThisFrame++;
                    }
                }

                // Draw if in frustum and mesh is ready
                if(inFrustum && c.meshes[s].vertexCount>0)
                    DrawMesh(c.meshes[s], sectionMat, MatrixIdentity());
            }
        }
    }

    // Project the 3D sun position to screen and run god rays toward it
    void DrawGodRays(RenderTexture2D& sceneTex, Camera3D& cam, int SW, int SH){
        // Place the sun very far away in the sun direction
        Vector3 sunWorld = Vector3Add(cam.position,
                           Vector3Scale(Vector3Normalize(sunDir), 400.0f));

        // Is sun in front of camera?
        // Dot goes from 1 (directly ahead) to -1 (directly behind).
        // Old code required dot > 0 for any rays, which cut them off at 90°
        // from center AND fully killed them unless sun was within 17° of center.
        // New code: full rays as long as sun is anywhere in the forward hemisphere
        // and fades softly as it moves behind the camera.
        Vector3 toCam = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
        Vector3 toSun = Vector3Normalize(Vector3Subtract(sunWorld,   cam.position));
        float   dot   = Vector3DotProduct(toCam, toSun);
        // Fade starts at dot=+0.2 (sun slightly off-axis) → full at dot=0.7
        float   vis   = Clamp((dot + 0.2f) / 0.7f, 0.0f, 1.0f);

        // Project to screen
        Vector2 sunScreen = GetWorldToScreen(sunWorld, cam);
        float sunU = sunScreen.x / (float)SW;
        float sunV = sunScreen.y / (float)SH;

        // Soft edge fade — rays persist even when sun is off-screen, just attenuate
        float edgeX = fabsf(sunU - 0.5f) * 2.0f;
        float edgeY = fabsf(sunV - 0.5f) * 2.0f;
        float edge  = Clamp(1.6f - fmaxf(edgeX, edgeY) * 1.2f, 0.0f, 1.0f);
        vis *= edge;
        vis *= (1.0f - depthFade); // fade rays underground

        float suv[2] = {sunU, sunV};
        SetShaderValue(godRayShader, locSunUV,      suv,  SHADER_UNIFORM_VEC2);
        SetShaderValue(godRayShader, locSunVisible, &vis, SHADER_UNIFORM_FLOAT);

        BeginShaderMode(godRayShader);
            DrawTexturePro(sceneTex.texture,
                {0,0,(float)SW,-(float)SH},
                {0,0,(float)SW, (float)SH},
                {0,0}, 0.0f, WHITE);
        EndShaderMode();
    }

    // DDA voxel raycast — returns first solid block hit within maxDist
    RayHit Raycast(Vector3 origin, Vector3 dir, float maxDist){
        RayHit res;
        dir = Vector3Normalize(dir);
        int bx=(int)floor(origin.x), by=(int)floor(origin.y), bz=(int)floor(origin.z);
        float dx=fabsf(dir.x), dy=fabsf(dir.y), dz=fabsf(dir.z);
        int sx=(dir.x>0)?1:-1, sy=(dir.y>0)?1:-1, sz=(dir.z>0)?1:-1;
        float tMaxX=(dx>1e-6f)?((dir.x>0?(bx+1-origin.x):(origin.x-bx))/dx):1e30f;
        float tMaxY=(dy>1e-6f)?((dir.y>0?(by+1-origin.y):(origin.y-by))/dy):1e30f;
        float tMaxZ=(dz>1e-6f)?((dir.z>0?(bz+1-origin.z):(origin.z-bz))/dz):1e30f;
        float tDX=(dx>1e-6f)?(1.0f/dx):1e30f;
        float tDY=(dy>1e-6f)?(1.0f/dy):1e30f;
        float tDZ=(dz>1e-6f)?(1.0f/dz):1e30f;
        int face=0;
        float t=0;
        while(t<maxDist){
            if(tMaxX<tMaxY&&tMaxX<tMaxZ){ t=tMaxX; bx+=sx; tMaxX+=tDX; face=0; }
            else if(tMaxY<tMaxZ)         { t=tMaxY; by+=sy; tMaxY+=tDY; face=1; }
            else                         { t=tMaxZ; bz+=sz; tMaxZ+=tDZ; face=2; }
            if(t>maxDist) break;
            if(IsSolid(bx,by,bz)){
                res.hit=true; res.wx=bx; res.wy=by; res.wz=bz;
                if(face==0) res.nx=-sx;
                else if(face==1) res.ny=-sy;
                else res.nz=-sz;
                return res;
            }
        }
        return res;
    }

    // Wireframe outline + optional break-progress darkening
    void DrawBlockHighlight(const RayHit& hit, float progress=0.0f){
        if(!hit.hit) return;
        float e=0.005f;
        Vector3 mn={(float)hit.wx-e,(float)hit.wy-e,(float)hit.wz-e};
        Vector3 mx={(float)hit.wx+1+e,(float)hit.wy+1+e,(float)hit.wz+1+e};
        Color col={255,255,255,180};
        DrawLine3D({mn.x,mn.y,mn.z},{mx.x,mn.y,mn.z},col);
        DrawLine3D({mx.x,mn.y,mn.z},{mx.x,mn.y,mx.z},col);
        DrawLine3D({mx.x,mn.y,mx.z},{mn.x,mn.y,mx.z},col);
        DrawLine3D({mn.x,mn.y,mx.z},{mn.x,mn.y,mn.z},col);
        DrawLine3D({mn.x,mx.y,mn.z},{mx.x,mx.y,mn.z},col);
        DrawLine3D({mx.x,mx.y,mn.z},{mx.x,mx.y,mx.z},col);
        DrawLine3D({mx.x,mx.y,mx.z},{mn.x,mx.y,mx.z},col);
        DrawLine3D({mn.x,mx.y,mx.z},{mn.x,mx.y,mn.z},col);
        DrawLine3D({mn.x,mn.y,mn.z},{mn.x,mx.y,mn.z},col);
        DrawLine3D({mx.x,mn.y,mn.z},{mx.x,mx.y,mn.z},col);
        DrawLine3D({mx.x,mn.y,mx.z},{mx.x,mx.y,mx.z},col);
        DrawLine3D({mn.x,mn.y,mx.z},{mn.x,mx.y,mx.z},col);

        // Progress: draw shrinking inner box as block cracks
        if(progress > 0.0f){
            float s2 = 0.5f - progress * 0.5f;   // inner box shrinks to centre
            float s3 = 0.5f;
            unsigned char alpha = (unsigned char)(120 + progress*100);
            Color pc = {255, (unsigned char)(255-progress*200), 50, alpha};
            float cx2=(mn.x+mx.x)*0.5f, cy2=(mn.y+mx.y)*0.5f, cz2=(mn.z+mx.z)*0.5f;
            Vector3 imn={cx2-s2,cy2-s2,cz2-s2};
            Vector3 imx={cx2+s2,cy2+s2,cz2+s2};
            // inner wireframe
            DrawLine3D({imn.x,imn.y,imn.z},{imx.x,imn.y,imn.z},pc);
            DrawLine3D({imx.x,imn.y,imn.z},{imx.x,imn.y,imx.z},pc);
            DrawLine3D({imx.x,imn.y,imx.z},{imn.x,imn.y,imx.z},pc);
            DrawLine3D({imn.x,imn.y,imx.z},{imn.x,imn.y,imn.z},pc);
            DrawLine3D({imn.x,imx.y,imn.z},{imx.x,imx.y,imn.z},pc);
            DrawLine3D({imx.x,imx.y,imn.z},{imx.x,imx.y,imx.z},pc);
            DrawLine3D({imx.x,imx.y,imx.z},{imn.x,imx.y,imx.z},pc);
            DrawLine3D({imn.x,imx.y,imx.z},{imn.x,imx.y,imn.z},pc);
            DrawLine3D({imn.x,imn.y,imn.z},{imn.x,imx.y,imn.z},pc);
            DrawLine3D({imx.x,imn.y,imn.z},{imx.x,imx.y,imn.z},pc);
            DrawLine3D({imx.x,imn.y,imx.z},{imx.x,imx.y,imx.z},pc);
            DrawLine3D({imn.x,imn.y,imx.z},{imn.x,imx.y,imx.z},pc);
        }
    }

    void Unload(){
        // Shut down both worker pools cleanly
        { std::lock_guard<std::mutex> lk(pendingMutex); stopWorkers=true; }
        pendingCV.notify_all();   // wake gen workers
        meshJobCV.notify_all();   // wake mesh workers
        for(auto& t:workers) if(t.joinable()) t.join();
        UnloadMaterial(sectionMat);
        UnloadShader(worldShader); UnloadShader(skyShader); UnloadShader(godRayShader);
        UnloadTexture(flameTex);
        if(torchModelLoaded) UnloadModel(torchModel);
        UnloadModel(skyModel); UnloadTexture(atlas); UnloadTexture(guiTex);
    }
};

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------
struct Player {
    Vector3 pos={24,465,24},vel={0,0,0}; // overridden at startup by SurfaceY lookup
    float yaw=-90,pitch=0,speed=6,jumpForce=8,gravity=-22,sens=0.12f;
    bool onGround=false;
    static constexpr float W=0.3f,H=1.8f,EYE=1.62f;

    Vector3 Forward() const{
        float y=yaw*DEG2RAD,p=pitch*DEG2RAD;
        return Vector3Normalize({cosf(p)*cosf(y),sinf(p),cosf(p)*sinf(y)});
    }
    Vector3 Right() const{return Vector3Normalize(Vector3CrossProduct(Forward(),{0,1,0}));}

    void ResolveAxis(World& w,Vector3& np,int axis){
        for(float dx:{-W,W}) for(float dz:{-W,W}) for(float dy=0;dy<=H;dy+=0.6f){
            if(w.IsSolid((int)floor(np.x+dx),(int)floor(np.y+dy),(int)floor(np.z+dz))){
                if(axis==0){np.x=pos.x;vel.x=0;}
                if(axis==1){if(vel.y<0)onGround=true;np.y=pos.y;vel.y=0;}
                if(axis==2){np.z=pos.z;vel.z=0;}
                return;
            }
        }
    }

    void Update(World& w,bool locked){
        if(!locked) return;
        float dt=GetFrameTime();
        Vector2 md=GetMouseDelta();
        yaw+=md.x*sens; pitch-=md.y*sens;
        pitch=Clamp(pitch,-89,89);
        Vector3 fwd  =Vector3Normalize({Forward().x,0,Forward().z});
        Vector3 right=Vector3Normalize({Right().x,0,Right().z});
        Vector3 move={0,0,0};
        if(IsKeyDown(KEY_W)) move=Vector3Add(move,fwd);
        if(IsKeyDown(KEY_S)) move=Vector3Add(move,Vector3Negate(fwd));
        if(IsKeyDown(KEY_D)) move=Vector3Add(move,right);
        if(IsKeyDown(KEY_A)) move=Vector3Add(move,Vector3Negate(right));
        if(Vector3Length(move)>0) move=Vector3Scale(Vector3Normalize(move),speed);
        vel.x=move.x; vel.z=move.z;
        vel.y+=gravity*dt;
        if(IsKeyDown(KEY_SPACE)&&onGround){vel.y=jumpForce;onGround=false;}
        onGround=false;
        Vector3 np=pos;
        np.x+=vel.x*dt; ResolveAxis(w,np,0); pos.x=np.x;
        np.y+=vel.y*dt; ResolveAxis(w,np,1); pos.y=np.y;
        np.z+=vel.z*dt; ResolveAxis(w,np,2); pos.z=np.z;
    }

    Camera3D ToRaylib() const{
        Vector3 eye={pos.x,pos.y+EYE,pos.z};
        Camera3D cam={}; cam.position=eye; cam.target=Vector3Add(eye,Forward());
        cam.up={0,1,0}; cam.fovy=75; cam.projection=CAMERA_PERSPECTIVE;
        return cam;
    }
};

// ---------------------------------------------------------------------------
// Menu
// ---------------------------------------------------------------------------
enum GameState { STATE_PLAYING, STATE_MENU, STATE_KEYBINDS, STATE_SETTINGS };

// A simple clickable button — returns true on left-click release
bool DrawMenuButton(int x, int y, int w, int h, const char* label,
                    Texture2D guiTex, bool hovered){
    Color fill   = hovered ? Color{58,48,28,240} : Color{28,24,16,220};
    Color border = hovered ? Color{220,180,55,255} : Color{90,80,58,180};
    Color tcol   = hovered ? Color{255,230,140,255} : Color{200,185,140,210};
    // Panel bg
    DrawNineSlice(guiTex, GuiRegions::panel, GuiRegions::panelBorder,
                  {(float)x,(float)y,(float)w,(float)h});
    // Tint overlay on hover
    if(hovered) DrawRectangle(x,y,w,h,{255,200,80,18});
    // Label
    int fs=18;
    int tw=MeasureText(label,fs);
    DrawText(label, x+1+w/2-tw/2, y+1+h/2-fs/2, fs, {0,0,0,120}); // shadow
    DrawText(label,   x+w/2-tw/2,   y+h/2-fs/2,   fs, tcol);
    Vector2 mp=GetMousePosition();
    bool over=(mp.x>=x&&mp.x<x+w&&mp.y>=y&&mp.y<y+h);
    return over && IsMouseButtonReleased(MOUSE_LEFT_BUTTON);
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
struct Settings {
    // FPS cap
    int   fpsOptions[5]  = {30,60,120,144,0}; // 0 = unlimited
    int   fpsIdx         = 1;                  // default 60

    // Resolution presets  (0 = don't force — use whatever window is)
    struct Res { int w,h; const char* label; };
    Res   resOptions[5]  = {{1280,720,"1280×720"},{1600,900,"1600×900"},
                             {1920,1080,"1920×1080"},{2560,1440,"2560×1440"},
                             {0,0,"Borderless FS"}};
    int   resIdx         = 0;

    // FOV
    int   fovOptions[5]  = {60,75,90,100,110};
    int   fovIdx         = 1;                  // default 75

    // Mouse sensitivity (stored as float × 100 so we can display without printf %f)
    float sensOptions[6] = {0.05f,0.08f,0.10f,0.12f,0.16f,0.20f};
    int   sensIdx        = 3;                  // default 0.12

    // God rays
    bool  godRays        = true;

    // Apply everything that can be changed instantly
    void Apply(int SW, int SH, Player& player){
        int cap = fpsOptions[fpsIdx];
        SetTargetFPS(cap == 0 ? 0 : cap);  // 0 = uncapped in raylib

        int ri = resIdx;
        if(resOptions[ri].w == 0){
            ToggleBorderlessWindowed();
        } else {
            SetWindowSize(resOptions[ri].w, resOptions[ri].h);
        }

        player.sens = sensOptions[sensIdx];
    }

    // Returns the current FOV value (caller pushes it into cam.fovy each frame)
    float Fov() const { return (float)fovOptions[fovIdx]; }
    float Sens() const { return sensOptions[sensIdx]; }
};

// Draw a left/right cycling control.  Returns -1 (left), 0 (none), +1 (right).
static int DrawCycler(int x, int y, int w, int h, const char* label, const char* value,
                      Texture2D guiTex){
    // Background row
    DrawNineSlice(guiTex, GuiRegions::panel, GuiRegions::panelBorder,
                  {(float)x,(float)y,(float)w,(float)h});

    // Label on the left
    int lfs=15;
    DrawText(label, x+10, y+h/2-lfs/2, lfs, {200,190,160,220});

    // Arrow buttons
    const int AW=24;
    Vector2 mp=GetMousePosition();
    bool lhov=(mp.x>=x+w/2     && mp.x<x+w/2+AW   && mp.y>=y && mp.y<y+h);
    bool rhov=(mp.x>=x+w-AW-8  && mp.x<x+w-8       && mp.y>=y && mp.y<y+h);

    Color ac={160,145,110,200}, ahov={255,210,80,255};
    // Left arrow
    DrawText("<", x+w/2,     y+h/2-lfs/2, lfs, lhov?ahov:ac);
    // Value in the middle
    int vw=MeasureText(value,lfs);
    DrawText(value, x+w/2+AW+(w/2-AW-AW-8-vw)/2, y+h/2-lfs/2, lfs, {240,225,175,255});
    // Right arrow
    DrawText(">", x+w-AW-8, y+h/2-lfs/2, lfs, rhov?ahov:ac);

    if(IsMouseButtonReleased(MOUSE_LEFT_BUTTON)){
        if(lhov) return -1;
        if(rhov) return +1;
    }
    return 0;
}

// Draw a toggle row.  Returns true if the toggle was clicked.
static bool DrawToggleRow(int x, int y, int w, int h, const char* label, bool on,
                          Texture2D guiTex){
    DrawNineSlice(guiTex, GuiRegions::panel, GuiRegions::panelBorder,
                  {(float)x,(float)y,(float)w,(float)h});
    int lfs=15;
    DrawText(label, x+10, y+h/2-lfs/2, lfs, {200,190,160,220});

    const char* val = on ? "ON" : "OFF";
    Color vc = on ? Color{100,220,100,255} : Color{200,80,80,255};
    int vw=MeasureText(val,lfs);
    DrawText(val, x+w-vw-14, y+h/2-lfs/2, lfs, vc);

    Vector2 mp=GetMousePosition();
    bool hov=(mp.x>=x && mp.x<x+w && mp.y>=y && mp.y<y+h);
    if(hov) DrawRectangle(x,y,w,h,{255,200,80,10});
    return hov && IsMouseButtonReleased(MOUSE_LEFT_BUTTON);
}
int main(){
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1280,720,"Delve");
    SetTargetFPS(60);
    SetWindowMinSize(640,360);
    SetExitKey(KEY_NULL); // disable ESC closing the window — we handle it ourselves

    bool cursorLocked=true;
    DisableCursor();

    World          world; world.Init();
    Player         player;
    ParticleSystem particles;
    Inventory      inventory;
    GameState      gameState = STATE_PLAYING;
    Settings       settings;
    player.sens = settings.Sens(); // apply default sensitivity
    inventory.Add(BLOCK_TORCH); // give player starting torches
    for(int i=0;i<15;i++) inventory.Add(BLOCK_TORCH); // 16 total

    // Generate spawn area synchronously so surface Y is known before first frame
    {
        int spawnCX=(int)floor(24.0f/CHUNK_SIZE);
        int spawnCZ=(int)floor(24.0f/CHUNK_SIZE);
        world.GenerateImmediate(spawnCX, spawnCZ, 1);
        int surfY = world.SurfaceY(24, 24);
        player.pos = {24.0f, (float)(surfY + 2), 24.0f};
    }

    int SW=GetScreenWidth(), SH=GetScreenHeight();
    RenderTexture2D sceneTex=LoadRenderTexture(SW,SH);

    // Breaking state
    float breakProgress = 0.0f;  // 0..1
    RayHit breakTarget;          // block being mined

    while(!WindowShouldClose()){
        float dt = GetFrameTime();

        // Window resize — recreate render texture if needed
        if(IsWindowResized()){
            SW=GetScreenWidth(); SH=GetScreenHeight();
            UnloadRenderTexture(sceneTex);
            sceneTex=LoadRenderTexture(SW,SH);
        }
        // F11 — toggle borderless fullscreen
        if(IsKeyPressed(KEY_F11)) ToggleBorderlessWindowed();

        // ESC — open/close menu (unless inventory is open, where E handles it)
        if(IsKeyPressed(KEY_ESCAPE)){
            if(gameState==STATE_PLAYING && !inventory.open){
                gameState=STATE_MENU;
                EnableCursor(); cursorLocked=false;
            } else if(gameState==STATE_MENU){
                gameState=STATE_PLAYING;
                DisableCursor(); cursorLocked=true;
            } else if(gameState==STATE_KEYBINDS || gameState==STATE_SETTINGS){
                gameState=STATE_MENU;
                EnableCursor(); cursorLocked=false;
            }
        }

        bool inGame = (gameState==STATE_PLAYING);
        player.Update(world, inGame && cursorLocked);
        world.Update(player.pos);
        world.UpdateTime(dt);
        world.UpdateDepth(player.pos.y, player.pos.x, player.pos.z);
        world.UpdateCamPos(player.pos);
        world.UpdateTorchLights(player.pos);
        Camera3D cam=player.ToRaylib();
        cam.fovy = settings.Fov();

        inventory.HandleInput(cursorLocked);
        if(inGame) inventory.HandleClick(SW,SH);

        // Raycast every frame
        RayHit hit;
        if(inGame && cursorLocked && !inventory.open)
            hit=world.Raycast(cam.position, player.Forward(), 6.0f);

        // --- Block breaking (hold LMB) ---
        if(inGame && cursorLocked && hit.hit && IsMouseButtonDown(MOUSE_LEFT_BUTTON)){
            // Reset progress if we switched target
            if(!breakTarget.hit ||
               breakTarget.wx!=hit.wx ||
               breakTarget.wy!=hit.wy ||
               breakTarget.wz!=hit.wz){
                breakProgress = 0.0f;
                breakTarget   = hit;
            }
            BlockType bt = world.GetBlock(hit.wx,hit.wy,hit.wz);
            if(BlockHardness(bt) > 1000.0f) { breakProgress=0.0f; breakTarget={}; } // bedrock
            else breakProgress += dt / BlockHardness(bt);
            if(breakProgress >= 1.0f){
                particles.Spawn({(float)hit.wx,(float)hit.wy,(float)hit.wz}, bt);
                inventory.Add(bt);
                world.SetBlock(hit.wx,hit.wy,hit.wz,BLOCK_AIR);
                breakProgress = 0.0f;
                breakTarget   = {};
            }
        } else {
            breakProgress = 0.0f;
            breakTarget   = {};
        }

        // --- Block placing (RMB) ---
        if(inGame && cursorLocked && hit.hit && IsMouseButtonPressed(MOUSE_RIGHT_BUTTON)){
            ItemStack& sel = inventory.slots[inventory.selected];
            if(sel.type != BLOCK_AIR && sel.count > 0){
                int px=hit.wx+hit.nx, py=hit.wy+hit.ny, pz=hit.wz+hit.nz;
                bool canPlace = world.GetBlock(px,py,pz)==BLOCK_AIR;
                // Torches need a solid block behind the hit face
                if(sel.type==BLOCK_TORCH){
                    canPlace = canPlace && world.IsSolid(hit.wx,hit.wy,hit.wz);
                    // Torches are non-solid — skip player overlap check
                    if(canPlace){
                        world.SetBlock(px,py,pz,sel.type);
                        inventory.Consume(inventory.selected);
                    }
                } else {
                    // Normal solid block — check player overlap
                    Vector3 pMin={player.pos.x-player.W, player.pos.y,          player.pos.z-player.W};
                    Vector3 pMax={player.pos.x+player.W, player.pos.y+player.H, player.pos.z+player.W};
                    bool overlap=(px+1>pMin.x&&px<pMax.x&&
                                  py+1>pMin.y&&py<pMax.y&&
                                  pz+1>pMin.z&&pz<pMax.z);
                    if(!overlap && canPlace){
                        world.SetBlock(px,py,pz,sel.type);
                        inventory.Consume(inventory.selected);
                    }
                }
            }
        }

        particles.Update(dt);

        // Render scene to texture
        BeginTextureMode(sceneTex);
            ClearBackground({10,8,18,255});
            BeginMode3D(cam);
                world.DrawSky({cam.position.x,cam.position.y,cam.position.z});
                world.DrawTerrain(player.pos.y, cam, SW, SH);
                world.DrawTorches(cam, (float)GetTime());
                world.DrawBlockHighlight(hit, breakProgress);
                particles.Draw(cam);
            EndMode3D();
        EndTextureMode();

        // Post-process + present
        BeginDrawing();
            ClearBackground(BLACK);
            if(settings.godRays)
                world.DrawGodRays(sceneTex,cam,SW,SH);
            else {
                // Just blit the scene without post-processing
                DrawTexturePro(sceneTex.texture,
                    {0,0,(float)SW,-(float)SH},{0,0,(float)SW,(float)SH},
                    {0,0},0.0f,WHITE);
            }

            if(gameState==STATE_PLAYING){
                // Crosshair
                int cx=SW/2,cy=SH/2;
                DrawLine(cx-10,cy,cx+10,cy,{255,255,255,160});
                DrawLine(cx,cy-10,cx,cy+10,{255,255,255,160});

                // Break progress bar
                if(breakProgress > 0.0f){
                    const int BW=80, BH=7;
                    int bx=cx-BW/2, by=cy+18;
                    DrawRectangle(bx-1,by-1,BW+2,BH+2,{0,0,0,160});
                    DrawRectangle(bx,by,BW,BH,{40,35,25,200});
                    int filled=(int)(breakProgress*BW);
                    unsigned char r=(unsigned char)(80+breakProgress*170);
                    unsigned char g=(unsigned char)(180-breakProgress*160);
                    DrawRectangle(bx,by,filled,BH,{r,g,30,230});
                    DrawRectangleLines(bx,by,BW,BH,{150,135,100,180});
                }
                inventory.Draw(SW,SH,world.atlas,world.guiTex);
                inventory.DrawHeld(world.atlas);
            }

            // ── Pause Menu ────────────────────────────────────────────────────
            if(gameState==STATE_MENU || gameState==STATE_KEYBINDS || gameState==STATE_SETTINGS){
                // Darken background
                DrawRectangle(0,0,SW,SH,{0,0,0,140});

                if(gameState==STATE_MENU){
                    const int MW=260, BH=46, BGAP=10, NBTN=4;
                    const int MH=60 + NBTN*(BH+BGAP) - BGAP + 20;
                    int mx=SW/2-MW/2, my=SH/2-MH/2;

                    DrawNineSlice(world.guiTex, GuiRegions::panel, GuiRegions::panelBorder,
                                  {(float)mx,(float)my,(float)MW,(float)MH});

                    const char* title="PAUSED"; int tfs=24;
                    DrawText(title, mx+1+MW/2-MeasureText(title,tfs)/2, my+1+16, tfs, {0,0,0,140});
                    DrawText(title,   mx+MW/2-MeasureText(title,tfs)/2,   my+16,   tfs, {220,200,140,255});

                    Vector2 mp=GetMousePosition();
                    int by0=my+60, bx=mx+20, bw=MW-40;
                    struct Btn{ const char* lbl; } btns[]={"Resume","Settings","Keybinds","Quit Game"};
                    for(int i=0;i<NBTN;i++){
                        int by_=by0+i*(BH+BGAP);
                        bool hov=(mp.x>=bx&&mp.x<bx+bw&&mp.y>=by_&&mp.y<by_+BH);
                        bool clicked=DrawMenuButton(bx,by_,bw,BH,btns[i].lbl,world.guiTex,hov);
                        if(clicked){
                            if(i==0){ gameState=STATE_PLAYING; DisableCursor(); cursorLocked=true; break; }
                            if(i==1){ gameState=STATE_SETTINGS; break; }
                            if(i==2){ gameState=STATE_KEYBINDS; break; }
                            if(i==3){ CloseWindow(); }
                        }
                    }
                    DrawText("[ESC] Resume", mx+MW/2-MeasureText("[ESC] Resume",11)/2,
                             my+MH-16, 11, {120,108,80,180});
                }
                else if(gameState==STATE_KEYBINDS){
                    const int KW=340, KH=360;
                    int kx=SW/2-KW/2, ky=SH/2-KH/2;
                    DrawNineSlice(world.guiTex, GuiRegions::panel, GuiRegions::panelBorder,
                                  {(float)kx,(float)ky,(float)KW,(float)KH});

                    const char* kt="KEYBINDS"; int tfs=20;
                    DrawText(kt, kx+1+KW/2-MeasureText(kt,tfs)/2, ky+1+14, tfs, {0,0,0,140});
                    DrawText(kt,   kx+KW/2-MeasureText(kt,tfs)/2,   ky+14,   tfs, {220,200,140,255});
                    DrawLine(kx+16, ky+42, kx+KW-16, ky+42, {100,90,65,160});

                    struct KBind{ const char* key; const char* action; };
                    KBind binds[]={
                        {"WASD",       "Move"},
                        {"Space",      "Jump"},
                        {"Mouse",      "Look"},
                        {"LMB (hold)", "Mine block"},
                        {"RMB",        "Place block"},
                        {"Scroll",     "Select hotbar slot"},
                        {"1 - 9",      "Hotbar slot"},
                        {"E",          "Open inventory"},
                        {"ESC",        "Pause menu"},
                        {"F11",        "Borderless fullscreen"},
                    };
                    int n=(int)(sizeof(binds)/sizeof(binds[0]));
                    int rowH=24, startY=ky+54;
                    for(int i=0;i<n;i++){
                        int ry=startY+i*rowH;
                        Color rowBg=(i%2==0)?Color{255,255,255,8}:Color{0,0,0,0};
                        DrawRectangle(kx+12,ry,KW-24,rowH-2,rowBg);
                        DrawText(binds[i].key,    kx+20,     ry+4, 14, {230,210,150,230});
                        DrawText(binds[i].action, kx+KW/2+4, ry+4, 14, {180,170,145,210});
                    }
                    DrawLine(kx+KW/2, ky+50, kx+KW/2, ky+50+n*rowH, {80,72,52,120});

                    Vector2 mp=GetMousePosition();
                    int bbx=kx+KW/2-60, bby=ky+KH-48, bbw=120, bbh=34;
                    bool hov=(mp.x>=bbx&&mp.x<bbx+bbw&&mp.y>=bby&&mp.y<bby+bbh);
                    if(DrawMenuButton(bbx,bby,bbw,bbh,"Back",world.guiTex,hov)){
                        gameState=STATE_MENU;
                    }
                    DrawText("[ESC] Back", kx+KW/2-MeasureText("[ESC] Back",11)/2,
                             ky+KH-12, 11, {120,108,80,180});
                }
                else if(gameState==STATE_SETTINGS){
                    // ── Settings panel ─────────────────────────────────────────
                    const int SW2=400, ROW=38, GAP=6, PAD=16;
                    const int NROWS=6; // FPS, Resolution, FOV, Sensitivity, God Rays, (apply btn)
                    const int SH2 = 50 + NROWS*(ROW+GAP) - GAP + 50;
                    int sx=SW/2-SW2/2, sy=SH/2-SH2/2;

                    DrawNineSlice(world.guiTex, GuiRegions::panel, GuiRegions::panelBorder,
                                  {(float)sx,(float)sy,(float)SW2,(float)SH2});

                    const char* st="SETTINGS"; int tfs=20;
                    DrawText(st, sx+1+SW2/2-MeasureText(st,tfs)/2, sy+1+14, tfs, {0,0,0,140});
                    DrawText(st,   sx+SW2/2-MeasureText(st,tfs)/2,   sy+14,   tfs, {220,200,140,255});
                    DrawLine(sx+16, sy+42, sx+SW2-16, sy+42, {100,90,65,160});

                    int rx=sx+PAD, rw=SW2-PAD*2, ry=sy+50;

                    // ── FPS Cap ──────────────────────────────────────────────
                    {
                        int cap=settings.fpsOptions[settings.fpsIdx];
                        char buf[16]; if(cap==0) TextCopy(buf,"Unlimited"); else sprintf(buf,"%d",cap);
                        int d=DrawCycler(rx,ry,rw,ROW,"FPS Cap",buf,world.guiTex);
                        if(d!=0){
                            settings.fpsIdx=(settings.fpsIdx+d+5)%5;
                            int c=settings.fpsOptions[settings.fpsIdx];
                            SetTargetFPS(c==0?0:c);
                        }
                        ry+=ROW+GAP;
                    }
                    // ── Resolution ───────────────────────────────────────────
                    {
                        int d=DrawCycler(rx,ry,rw,ROW,"Resolution",
                                         settings.resOptions[settings.resIdx].label,world.guiTex);
                        if(d!=0){
                            settings.resIdx=(settings.resIdx+d+5)%5;
                            auto& r=settings.resOptions[settings.resIdx];
                            if(r.w==0){
                                ToggleBorderlessWindowed();
                            } else {
                                SetWindowSize(r.w,r.h);
                            }
                        }
                        ry+=ROW+GAP;
                    }
                    // ── FOV ──────────────────────────────────────────────────
                    {
                        char buf[8]; sprintf(buf,"%d",settings.fovOptions[settings.fovIdx]);
                        int d=DrawCycler(rx,ry,rw,ROW,"Field of View",buf,world.guiTex);
                        if(d!=0) settings.fovIdx=(settings.fovIdx+d+5)%5;
                        ry+=ROW+GAP;
                    }
                    // ── Mouse Sensitivity ────────────────────────────────────
                    {
                        // Display as integer percentage (0.12 → "12")
                        char buf[8];
                        sprintf(buf,"%d",(int)(settings.sensOptions[settings.sensIdx]*100));
                        int d=DrawCycler(rx,ry,rw,ROW,"Mouse Sensitivity",buf,world.guiTex);
                        if(d!=0){
                            settings.sensIdx=(settings.sensIdx+d+6)%6;
                            player.sens=settings.Sens();
                        }
                        ry+=ROW+GAP;
                    }
                    // ── God Rays ─────────────────────────────────────────────
                    {
                        if(DrawToggleRow(rx,ry,rw,ROW,"God Rays",settings.godRays,world.guiTex))
                            settings.godRays=!settings.godRays;
                        ry+=ROW+GAP;
                    }

                    // ── Back button ──────────────────────────────────────────
                    {
                        Vector2 mp=GetMousePosition();
                        int bbx=sx+SW2/2-60, bby=sy+SH2-46, bbw=120, bbh=34;
                        bool hov=(mp.x>=bbx&&mp.x<bbx+bbw&&mp.y>=bby&&mp.y<bby+bbh);
                        if(DrawMenuButton(bbx,bby,bbw,bbh,"Back",world.guiTex,hov))
                            gameState=STATE_MENU;
                        DrawText("[ESC] Back", sx+SW2/2-MeasureText("[ESC] Back",11)/2,
                                 sy+SH2-14, 11, {120,108,80,180});
                    }
                }
            }

            DrawFPS(10,10);
        EndDrawing();
    }

    UnloadRenderTexture(sceneTex);
    world.Unload();
    CloseWindow();
    return 0;
}