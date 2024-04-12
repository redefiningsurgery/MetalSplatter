#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

constant const int kMaxViewCount = 2;
constant static const half kBoundsRadius = 2;
constant static const half kBoundsRadiusSquared = kBoundsRadius*kBoundsRadius;

enum BufferIndex: int32_t
{
    BufferIndexUniforms = 0,
    BufferIndexSplat    = 1,
};

typedef struct
{
    matrix_float4x4 projectionMatrix;
    matrix_float4x4 viewMatrix;
    uint2 screenSize;
    float time;
} Uniforms;

typedef struct
{
    Uniforms uniforms[kMaxViewCount];
} UniformsArray;

typedef struct
{
    packed_float3 position;
    packed_half4 color;
//    packed_half3 covA;
//    packed_half3 covB;
    packed_half2 quaternions0;
    packed_half2 quaternions1;
    packed_half2 scale0;
    packed_half2 scale1;
    packed_half2 motion0;
    packed_half2 motion1;
    packed_half2 motion2;
    packed_half2 motion3;
    packed_half2 motion4;
    packed_half2 rotation0;
    packed_half2 rotation1;
    packed_half2 rbf;
} Splat;

typedef struct
{
    float4 position [[position]];
    half2 relativePosition; // Ranges from -kBoundsRadius to +kBoundsRadius
    half4 color;
} ColorInOut;

//float3 calcCovariance2D(float3 viewPos,
//                        packed_half3 cov3Da,
//                        packed_half3 cov3Db,
//                        float4x4 viewMatrix,
//                        float4x4 projectionMatrix,
//                        uint2 screenSize) {
//    
//}

// cov2D is a flattened 2d covariance matrix. Given
// covariance = | a b |
//              | c d |
// (where b == c because the Gaussian covariance matrix is symmetric),
// cov2D = ( a, b, d )
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b; // matrix is symmetric, so "c" is same as "b"
    float trace = a + d;

    float mean = 0.5 * trace;
    float dist = max(0.1, sqrt(mean * mean - det)); // based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu

    // Eigenvalues
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;

    float2 eigenvector1;
    if (b == 0) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        eigenvector1 = normalize(float2(b, d - lambda2));
    }

    // Gaussian axes are orthogonal
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);

    lambda1 *= 2;
    lambda2 *= 2;

    v1 = eigenvector1 * sqrt(lambda1);
    v2 = eigenvector2 * sqrt(lambda2);
}

vertex ColorInOut splatVertexShader(uint vertexID [[vertex_id]],
                                    uint instanceID [[instance_id]],
                                    ushort amp_id [[amplification_id]],
                                    constant Splat* splatArray [[ buffer(BufferIndexSplat) ]],
                                    constant UniformsArray & uniformsArray [[ buffer(BufferIndexUniforms) ]]) {
    ColorInOut out;

    Uniforms uniforms = uniformsArray.uniforms[min(int(amp_id), kMaxViewCount)];

    Splat splat = splatArray[instanceID];
    
    float time = uniforms.time;
    packed_half2 trbf = splat.rbf;
    float dt = time - trbf.x;
    
    float topacity = exp(-1.0 * pow(dt / trbf.y, 2.0));
    if (topacity < 0.02){
        return out;
    }
//    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
//                                    uniforms.viewMatrix, uniforms.projectionMatrix, uniforms.screenSize);
    
    packed_half2 m0 = splat.motion0;
    packed_half2 m1 = splat.motion1;
    packed_half2 m2 = splat.motion2;
    packed_half2 m3 = splat.motion3;
    packed_half2 m4 = splat.motion4;
    
    float4 trot = float4(splat.rotation0.x, splat.rotation0.y, splat.rotation1.x, splat.rotation1.y) * dt;
    float3 tpos = (float3(m0.x, m0.y, m1.x) * dt + float3(m1.y,m2.x, m2.y) * dt * dt + float3(m3.x, m3.y, m4.x) * dt * dt * dt);
    
    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position+tpos, 1);
    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;
    
// TODO: how to handle this
    float clip = 1.2 * projectedCenter.w;
    if (projectedCenter.z < -clip || projectedCenter.x < -clip || projectedCenter.x > clip || projectedCenter.y < -clip || projectedCenter.y > clip){
        return out;
    }
    
    float4 rot = float4(splat.quaternions0.x, splat.quaternions0.y, splat.quaternions1.x, splat.quaternions1.y) + trot;
    float rot_det = sqrt(dot(rot,rot));
    rot = rot / rot_det;
    float3x3 S = float3x3(
                  splat.scale0.x, 0.0, 0.0,
                  0.0, splat.scale0.y, 0.0,
                  0.0, 0.0, splat.scale1.x
                          );
    float3x3 R = float3x3(
                          1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]), 2.0 * (rot[1] * rot[2] - rot[0] * rot[3]), 2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                          2.0 * (rot[1] * rot[2] + rot[0] * rot[3]), 1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]), 2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                          2.0 * (rot[1] * rot[3] - rot[0] * rot[2]), 2.0 * (rot[2] * rot[3] + rot[0] * rot[1]), 1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]));
    
    float3x3 M = S * R;
    float3x3 Vrk = 4.0 * transpose(M) * M;
    
    
    float4x4 viewMatrix = uniforms.viewMatrix;
    float4x4 projectionMatrix = uniforms.projectionMatrix;
    uint2 screenSize = uniforms.screenSize;
    
    float invViewPosZ = 1 / viewPosition4.z;
    float invViewPosZSquared = invViewPosZ * invViewPosZ;

//    float tanHalfFovX = 1 / projectionMatrix[0][0];
//    float tanHalfFovY = 1 / projectionMatrix[1][1];
//    float limX = 1.3 * tanHalfFovX;
//    float limY = 1.3 * tanHalfFovY;
//    viewPosition4.x = clamp(viewPosition4.x * invViewPosZ, -limX, limX) * viewPosition4.z;
//    viewPosition4.y = clamp(viewPosition4.y * invViewPosZ, -limY, limY) * viewPosition4.z;

    float focalX = screenSize.x * projectionMatrix[0][0] / 2;
    float focalY = screenSize.y * projectionMatrix[1][1] / 2;

    float3x3 J = float3x3(
            focalX * invViewPosZ, 0, 0,
            0, focalY * invViewPosZ, 0,
            -(focalX * viewPosition4.x) * invViewPosZSquared, -(focalY * viewPosition4.y) * invViewPosZSquared, 0
        );
    
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    
    float3x3 cov = T * Vrk * transpose(T);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    float3 cov2D =  float3(cov[0][0], cov[0][1], cov[1][1]);
    
    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    
    out.color = clamp(projectedCenter.z/projectedCenter.w+1.0, 0.0, 1.0) * float4(1.0,1.0,1.0,topacity) * float4(splat.color.xyzw);
    float2 vCenter = float2(projectedCenter)/projectedCenter.w;
//    TODO: Figure out position and relative position
    out.position = float4(vCenter+
                          )
    

//    float bounds = 1.2 * projectedCenter.w;
//    if (projectedCenter.z < -projectedCenter.w ||
//        projectedCenter.x < -bounds ||
//        projectedCenter.x > bounds ||
//        projectedCenter.y < -bounds ||
//        projectedCenter.y > bounds) {
//        out.position = float4(1, 1, 0, 1);
//        return out;
//    }

//    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
//    half2 relativeCoordinates = relativeCoordinatesArray[vertexID];
//    half2 screenSizeFloat = half2(uniforms.screenSize.x, uniforms.screenSize.y);
//    half2 projectedScreenDelta =
//        (relativeCoordinates.x * half2(axis1) + relativeCoordinates.y * half2(axis2))
//        * 2
//        * kBoundsRadius
//        / screenSizeFloat;

//    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
//                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
//                          projectedCenter.z,
//                          projectedCenter.w);
//    out.relativePosition = kBoundsRadius * relativeCoordinates;
//    out.color = splat.color;
    return out;
}

fragment half4 splatFragmentShader(ColorInOut in [[stage_in]]) {
    half2 v = in.relativePosition;
    half negativeVSquared = -dot(v, v);
    if (negativeVSquared < -kBoundsRadiusSquared) {
        discard_fragment();
    }

    half alpha = exp(negativeVSquared) * in.color.a;
    return half4(alpha * in.color.rgb, alpha);
}

