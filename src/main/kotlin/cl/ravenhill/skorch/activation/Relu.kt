package cl.ravenhill.skorch.activation

import cl.ravenhill.skorch.DefaultOps
import cl.ravenhill.skorch.toFloat
import org.jetbrains.kotlinx.dl.api.core.activation.Activation
import org.jetbrains.kotlinx.dl.api.inference.toFloatArray
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.MathOps
import org.tensorflow.op.Ops
import kotlin.system.exitProcess


/**
 * Defines the Rectified Linear Unit (ReLU) activation function class.
 *
 * The ReLU activation function outputs the input directly if it's positive; otherwise, it
 * outputs zero.
 * It's a piecewise linear function that will output the input directly if it is positive,
 * otherwise, it will output zero.
 *
 * This class provides methods for applying the ReLU function to a tensor.
 *
 * ## Examples
 * ### Example 1: Applying ReLU Activation to a Tensor
 * ```kotlin
 * val relu = Relu()
 * val features: Operand<Float> = ...
 * val activatedFeatures = relu.apply(DefaultOps.tf, features)
 * ```
 * ### Example 2: Using ReLU Activation in a Neural Network Layer
 * ```kotlin
 * val layer = DenseLayer(128, activation=Relu())
 * ```
 */
class Relu : Activation {

    /**
     * Applies the ReLU activation function to a given tensor of features.
     *
     * @param tf TensorFlow's operations reference.
     * @param features The tensor features to which the activation function will be applied.
     *
     * @return A tensor with the same shape as `features`, where the ReLU function has been applied
     * elementwise.
     */
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        with(tf) {
            with(math) {
                maximum(features, constant(0.0f))
            }
        }

    /**
     * Invokes the ReLU activation function on a given tensor of features.
     *
     * This function is shorthand for calling `apply(DefaultOps.tf, features).asOutput().tensor()`.
     *
     * @return A tensor with the same shape as `features`, where the ReLU function has been applied elementwise.
     */
    operator fun invoke(features: Operand<Float>): Tensor<Float> =
        apply(DefaultOps.tf, features).asOutput().tensor()

    /**
     * Invokes the ReLU activation function on a single float value.
     *
     * This function is shorthand for calling
     * `apply(DefaultOps.tf, constant(features)).asOutput().tensor().floatValue()`.
     *
     * @return The output of the ReLU function applied to the input float value.
     */
    operator fun invoke(features: Float): Float =
        invoke(DefaultOps.tf.constant(features)).floatValue()
}

fun main() {
    val relu = Relu()
    val features: Operand<Float> = DefaultOps.tf.constant(floatArrayOf(0.0f, 1.0f, 50.0f, 100.0f))
    println(relu(features).toFloatArray().toList())
    exitProcess(0)
}
