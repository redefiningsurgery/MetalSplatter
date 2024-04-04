import Foundation
import simd

public struct SplatScenePoint {
    public enum Color {
        case sphericalHarmonic(Float, Float, Float, [Float])
        case firstOrderSphericalHarmonic(Float, Float, Float)
        case linearFloat(Float, Float, Float)
        case linearUInt8(UInt8, UInt8, UInt8)
        case none

        var nonFirstOrderSphericalHarmonics: [Float]? {
            switch self {
            case .sphericalHarmonic(_, _, _, let nonFirstOrderSphericalHarmonics):
                nonFirstOrderSphericalHarmonics
            case .firstOrderSphericalHarmonic, .linearFloat, .linearUInt8:
                nil
            case .none:
                nil
            }
        }
    }

    public var position: SIMD3<Float>
    public var normal: SIMD3<Float>
    public var color: Color
    public var opacity: Float
    public var scale: SIMD3<Float>
    public var rotation: simd_quatf
// Add support for SpaceTimeGaussians
    public var motion: SIMD16<Float>
    public var omega: simd_quatf
    public var trbfCenter: Float
    public var trbfScale: Float

    public init(position: SIMD3<Float>,
                normal: SIMD3<Float>,
                color: Color,
                opacity: Float,
                scale: SIMD3<Float>,
                rotation: simd_quatf, //) {
                motion: SIMD16<Float>,
                omega: simd_quatf,
                trbfCenter:Float,
                trbfScale: Float){
        self.position = position
        self.normal = normal
        self.color = color
        self.opacity = opacity
        self.scale = scale
        self.rotation = rotation
        self.motion = motion
        self.omega = omega
        self.trbfCenter = trbfCenter
        self.trbfScale = trbfScale
    }
}
