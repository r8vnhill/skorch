package cl.ravenhill.skorch.activation

import cl.ravenhill.skorch.DefaultOps
import cl.ravenhill.skorch.plus
import cl.ravenhill.skorch.toFloat
import cl.ravenhill.skorch.unaryMinus
import org.jetbrains.kotlinx.dl.api.core.activation.Activation
import org.jetbrains.kotlinx.dl.api.inference.toFloatArray
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import kotlin.system.exitProcess

/**
 * Defines the Sigmoid activation function class, which is a common activation function in neural
 * networks.
 *
 * This class extends the [Activation] class and provides methods for applying the Sigmoid function
 * to a tensor.
 *
 * ## Examples
 * ### Example 1: Applying Sigmoid Activation to a Tensor
 * ```kotlin
 * val sigmoid = Sigmoid()
 * val features: Operand<Float> = ...
 * val activatedFeatures = sigmoid(features)
 * ```
 * ### Example 2: Using Sigmoid Activation in a Neural Network Layer
 * ```kotlin
 * val layer = DenseLayer(128, activation=Sigmoid())
 * ```
 */
class Sigmoid : Activation {
    /**
     * Applies the Sigmoid activation function to a given tensor of features.
     *
     * @return A tensor with the same shape as `features`, where the Sigmoid function has been
     * applied elementwise.
     */
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        with(tf) {
            with(math) {
                reciprocal(1f + exp(-features))
            }
        }

    /**
     * Invokes the Sigmoid activation function on a given tensor of features.
     *
     * This function is shorthand for calling `apply(DefaultOps.tf, features).asOutput().tensor()`.
     *
     * @return A tensor with the same shape as `features`, where the Sigmoid function has been
     * applied elementwise.
     */
    operator fun invoke(features: Operand<Float>): Tensor<Float> =
        apply(DefaultOps.tf, features).asOutput().tensor()

    /**
     * Invokes the Sigmoid activation function on a single float value.
     *
     * This function is shorthand for calling
     * `apply(DefaultOps.tf, constant(float)).asOutput().tensor().toFloat()`.
     *
     * @return The output of the Sigmoid function applied to the input float value.
     */
    operator fun invoke(float: Float): Float =
        invoke(DefaultOps.tf.constant(float)).toFloat()
}

fun main() {
    val sigmoid = Sigmoid()
    val features1 = DefaultOps.tf.constant(floatArrayOf(0.0f, 1.0f, 50.0f, 100.0f))
    println(sigmoid(features1).toFloatArray().toList() == listOf(0.5f, 0.7310586f, 1.0f, 1.0f))
    val features2 = DefaultOps.tf.constant(floatArrayOf(-100.0f, -50.0f, -1.0f, 0.0f))
    println(
        sigmoid(features2).toFloatArray().toList() == listOf(
            0.0f,
            1.9287499E-22f,
            0.26894143f,
            0.5f
        )
    )
    exitProcess(0)
}