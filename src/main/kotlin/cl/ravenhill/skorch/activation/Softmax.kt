package cl.ravenhill.skorch.activation

import cl.ravenhill.skorch.DefaultOps
import cl.ravenhill.skorch.div
import cl.ravenhill.skorch.minus
import org.jetbrains.kotlinx.dl.api.core.activation.Activation
import org.jetbrains.kotlinx.dl.api.inference.toFloatArray
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Max
import org.tensorflow.op.core.Sum
import kotlin.system.exitProcess


/**
 * Defines the Softmax activation function class.
 *
 * The Softmax activation function is defined as:
 * ```
 * softmax(x) = exp(x) / sum(exp(x))
 * ```
 *
 * ## Examples
 * ### Example 1: Applying Softmax Activation to a Tensor
 * ```kotlin
 * val softmax = Softmax(0)
 * val features: Operand<Float> = ...
 * val probabilities = softmax.apply(DefaultOps.tf, features)
 * ```
 * ### Example 2: Using Softmax Activation in a Neural Network Layer
 * ```kotlin
 * val layer = DenseLayer(128, activation=Softmax(0))
 * ```
 *
 * ## Notes
 * The Softmax activation function is often used in the final layer of a neural network-based
 * classifier.
 * It converts a real vector to a vector of categorical probabilities, with elements in the range
 * (0, 1) that add up to 1.
 * This class provides methods for applying the Softmax function to a tensor, optionally making the
 * function numerically stable.
 *
 * @property axis The dimension softmax would be performed on.
 * @property stable A boolean flag indicating whether to make the function numerically stable.
 */
class Softmax(private val axis: Int, private val stable: Boolean = true) : Activation {
    /**
     * Applies the Softmax activation function to a given tensor of features.
     *
     * If `stable` is true, the function is numerically stabilized.
     * The stabilization subtracts the maximum of each element in the input tensor from each element
     * to avoid large exponentials and possible overflow.
     *
     * @param tf TensorFlow operations reference.
     * @param features The tensor features to which the activation function will be applied.
     *
     * @return A tensor with the same shape as `features`, where the Softmax function has been
     * applied along the specified axis.
     */
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> {
        with(tf) {
            with(math) {
                val e = exp(if (stable) {
                    features - max(features, constant(axis), Max.keepDims(true))
                } else {
                    features
                })
                return e / sum(e, constant(axis), Sum.keepDims(true))
            }
        }
    }

    /**
     * Invokes the Softmax activation function on a given tensor of features.
     *
     * This function is shorthand for calling `apply(DefaultOps.tf, features).asOutput().tensor()`.
     *
     * @return A tensor with the same shape as `features`, where the Softmax function has been
     * applied along the specified axis.
     */
    operator fun invoke(features: Operand<Float>): Tensor<Float> =
        apply(DefaultOps.tf, features).asOutput().tensor()

    /**
     * Invokes the Softmax activation function on a single float value.
     *
     * This function is shorthand for calling
     * `apply(DefaultOps.tf, constant(features)).asOutput().tensor().floatValue()`.
     *
     * @return The output of the Softmax function applied to the input float value.
     */
    operator fun invoke(features: Float): Float =
        invoke(DefaultOps.tf.constant(features)).floatValue()
}

fun main() {
    val tf = DefaultOps.tf
    val softmax = Softmax(0)
    val features = tf.constant(floatArrayOf(1f, 2f, 3f, 4f, 1f, 2f, 3f))
    val activatedFeatures = softmax(features)
    println(activatedFeatures.toFloatArray().contentToString())
    exitProcess(0)
}