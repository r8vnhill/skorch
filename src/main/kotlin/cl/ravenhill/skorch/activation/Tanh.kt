package cl.ravenhill.skorch.activation

import cl.ravenhill.skorch.DefaultOps
import cl.ravenhill.skorch.div
import cl.ravenhill.skorch.minus
import cl.ravenhill.skorch.plus
import cl.ravenhill.skorch.times
import cl.ravenhill.skorch.toFloat
import org.jetbrains.kotlinx.dl.api.core.activation.Activation
import org.jetbrains.kotlinx.dl.api.inference.toFloatArray
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import kotlin.system.exitProcess


/**
 * Defines the Hyperbolic Tangent (Tanh) activation function class, which is a common activation
 * function in neural networks.
 *
 * This class extends the [Activation] class and provides methods for applying the Tanh function to
 * a tensor.
 *
 * ## Examples
 * ### Example 1: Applying Tanh Activation to a Tensor
 * ```kotlin
 * val tanh = Tanh()
 * val features: Operand<Float> = ...
 * val activatedFeatures = tanh(features)
 * ```
 * ### Example 2: Using Tanh Activation in a Neural Network Layer
 * ```kotlin
 * val layer = DenseLayer(128, activation=Tanh())
 * ```
 */
class Tanh : Activation {

    /**
     * Applies the Tanh activation function to a given tensor of features.
     *
     * @return A tensor with the same shape as `features`, where the Tanh function has been applied
     * elementwise.
     */
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        with(tf) {
            with(math) {

                val e = exp(2f * features)
                val r = reciprocal(e)
                return (e - r) / (e + r)
            }
        }
    }

    /**
     * Invokes the Tanh activation function on a given tensor of features.
     *
     * This function is shorthand for calling `apply(defaultOps, features).asOutput().tensor()`.
     *
     * @return A tensor with the same shape as `features`, where the Tanh function has been applied
     * elementwise.
     */
    operator fun invoke(features: Operand<Float>): Tensor<Float> =
        apply(DefaultOps.tf, features).asOutput().tensor()

    /**
     * Invokes the Tanh activation function on a single float value.
     *
     * This function is shorthand for calling `apply(defaultOps, constant(float)).asOutput().tensor().toFloat()`.
     *
     * @return The output of the Tanh function applied to the input float value.
     */
    operator fun invoke(float: Float): Float =
        invoke(DefaultOps.tf.constant(float)).toFloat()
}

fun main() {
    val tanh = Tanh()
    val activatedFeatures = tanh(
        DefaultOps.tf.constant(
            floatArrayOf(
                -5f,
                -0.5f,
                1f,
                1.2f,
                2f,
                3f,
            )
        )
    )
    println(activatedFeatures.toFloatArray().toList())

    exitProcess(0)
}
