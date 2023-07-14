package cl.ravenhill.skorch.activation

import cl.ravenhill.skorch.DefaultOps
import cl.ravenhill.skorch.div
import cl.ravenhill.skorch.maximum
import cl.ravenhill.skorch.minimum
import cl.ravenhill.skorch.minus
import cl.ravenhill.skorch.plus
import cl.ravenhill.skorch.times
import org.jetbrains.kotlinx.dl.api.core.activation.Activation
import org.jetbrains.kotlinx.dl.api.inference.toFloatArray
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import kotlin.system.exitProcess


/**
 * Defines the Continuously Differentiable Exponential Linear Unit (CELU) activation function class.
 *
 * The CELU activation function generalizes the Exponential Linear Unit (ELU) activation function by
 * introducing an additional `alpha` parameter.
 * The function is defined as `max(0, x) + min(0, alpha * (exp(x / alpha) - 1))`.
 *
 * The `alpha` parameter controls the saturation level and to which extent negative inputs are
 * dampened.
 *
 * This class provides methods for applying the CELU function to a tensor.
 *
 * ## Examples
 * ### Example 1: Applying CELU Activation to a Tensor
 * ```kotlin
 * val celu = Celu(0.5f)
 * val features: Operand<Float> = ...
 * val activatedFeatures = celu.apply(DefaultOps.tf, features)
 * ```
 * ### Example 2: Using CELU Activation in a Neural Network Layer
 * ```kotlin
 * val layer = DenseLayer(128, activation=Celu(0.5f))
 * ```
 *
 * @property alpha The alpha parameter of the CELU function.
 */
class Celu(private val alpha: Float = 1f) : Activation {

    /**
     * Applies the CELU activation function to a given tensor of features.
     *
     * @param tf TensorFlow operations reference.
     * @param features The tensor features to which the activation function will be applied.
     *
     * @return A tensor with the same shape as `features`, where the CELU function has been applied
     * elementwise.
     */
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        with(tf) {
            with(math) {
                maximum(0f, features) + minimum(0f, alpha * (exp(features / alpha) - 1f))
            }
        }

    /**
     * Invokes the CELU activation function on a given tensor of features.
     *
     * This function is shorthand for calling `apply(DefaultOps.tf, features)`.
     *
     * @return A tensor with the same shape as `features`, where the CELU function has been applied
     * elementwise.
     */
    operator fun invoke(features: Operand<Float>): Operand<Float> =
        apply(DefaultOps.tf, features)

    /**
     * Invokes the CELU activation function on a single float value.
     *
     * This function is shorthand for calling
     * `apply(DefaultOps.tf, constant(features)).asOutput().tensor().floatValue()`.
     *
     * @return The output of the CELU function applied to the input float value.
     */
    operator fun invoke(features: Float): Float =
        invoke(DefaultOps.tf.constant(features)).asOutput().tensor().floatValue()
}

fun main() {
    val celu = Celu()
    val features: Operand<Float> = DefaultOps.tf.constant(floatArrayOf(-1f, 0f, 1f, 2f, 3f))
    val activatedFeatures = celu(features)
    println(activatedFeatures.asOutput().tensor().toFloatArray().contentToString())
    exitProcess(0)
}
