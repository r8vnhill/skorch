package cl.ravenhill.skorch.activation

import cl.ravenhill.skorch.DefaultOps
import cl.ravenhill.skorch.times
import org.jetbrains.kotlinx.dl.api.core.activation.Activation
import org.jetbrains.kotlinx.dl.api.inference.toFloatArray
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import kotlin.system.exitProcess


/**
 * Defines the Swish activation function class.
 *
 * The Swish activation function computes the elementwise value of `x * sigmoid(x)`.
 * It tends to work better than ReLU on deeper models across a number of challenging data sets.
 *
 * This class provides methods for applying the Swish function to a tensor.
 *
 * ## Examples
 * ### Example 1: Applying Swish Activation to a Tensor
 * ```kotlin
 * val swish = Swish()
 * val features: Operand<Float> = ...
 * val activatedFeatures = swish.apply(DefaultOps.tf, features)
 * ```
 * ### Example 2: Using Swish Activation in a Neural Network Layer
 * ```kotlin
 * val layer = DenseLayer(128, activation=Swish())
 * ```
 */
class Swish(private val beta: Float = 1.0f) : Activation {

    /**
     * Applies the Swish activation function to a given tensor of features.
     *
     * @param tf TensorFlow operations reference.
     * @param features The tensor features to which the activation function will be applied.
     *
     * @return A tensor with the same shape as `features`, where the Swish function has been applied elementwise.
     */
    override fun apply(tf: Ops, features: Operand<Float>): Operand<Float> =
        with(tf) {
            with(math) {
                features * Sigmoid().apply(tf, beta * features)
            }
        }

    /**
     * Invokes the Swish activation function on a given tensor of features.
     *
     * This function is shorthand for calling `apply(DefaultOps.tf, features)`.
     *
     * @return A tensor with the same shape as `features`, where the Swish function has been applied elementwise.
     */
    operator fun invoke(features: Operand<Float>): Operand<Float> =
        apply(DefaultOps.tf, features)

    /**
     * Invokes the Swish activation function on a single float value.
     *
     * This function is shorthand for calling `apply(DefaultOps.tf, constant(features)).asOutput().tensor().floatValue()`.
     *
     * @return The output of the Swish function applied to the input float value.
     */
    operator fun invoke(features: Float): Float =
        invoke(DefaultOps.tf.constant(features)).asOutput().tensor().floatValue()
}

fun main() {
    val swish = Swish()
    val features: Operand<Float> = DefaultOps.tf.constant(floatArrayOf(1f, 2f, 3f, 4f))
    val activatedFeatures = swish(features)
    println(activatedFeatures.asOutput().tensor().toFloatArray().contentToString())
    exitProcess(0)
}