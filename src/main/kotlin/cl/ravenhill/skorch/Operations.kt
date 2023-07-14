package cl.ravenhill.skorch

import cl.ravenhill.skorch.DefaultOps.tf
import org.tensorflow.Operand
import org.tensorflow.Tensor
import org.tensorflow.op.MathOps
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Div
import org.tensorflow.op.math.Mul
import org.tensorflow.op.math.Sub


/**
 * Singleton object that holds the default TensorFlow operations.
 *
 * @property tf The default TensorFlow operations, initialized with [Ops.create] with no arguments.
 */
object DefaultOps {
    var tf: Ops = Ops.create()
}

/**
 * Defines the addition operation for two operands in the context of TensorFlow operations and math
 * operations.
 *
 * @param right The operand to be added to the current operand.
 * @return An operand which represents the result of the addition.
 */
context(Ops, MathOps)
operator fun <T> Operand<T>.plus(right: Operand<T>): Operand<T> = add(this, right)

/**
 * Defines the subtraction operation for two operands in the context of TensorFlow operations and
 * math operations.
 *
 * @param right The operand to be subtracted from the current operand.
 * @return An operand which represents the result of the subtraction.
 */
context(Ops, MathOps)
operator fun <T> Operand<T>.minus(right: Operand<T>): Sub<T> = sub(this, right)

/**
 * Overloads the subtraction operator for a tensor and a float.
 *
 * This extension function provides a more readable syntax for subtracting a float from a TensorFlow
 * operand (tensor).
 * It wraps around the `sub` operation provided by TensorFlow and returns the result as a new
 * tensor.
 *
 * @receiver The tensor operand in the subtraction.
 * @param right The float value to be subtracted from the tensor.
 *
 * @return The result of the subtraction operation as a `Sub` object.
 */
context (Ops, MathOps)
operator fun Operand<Float>.minus(right: Float): Sub<Float> = sub(this, constant(right))

/**
 * Defines the multiplication operation for a Float scalar and an operand in the context of
 * TensorFlow operations and math operations.
 *
 * @param factor The operand to be multiplied with the Float scalar.
 * @return An operand which represents the result of the multiplication.
 */
context(Ops, MathOps)
operator fun Float.times(factor: Operand<Float>): Operand<Float> =
    mul(constant(this), factor)

/**
 * Overloads the multiplication operator for a pair of operands of the same type.
 *
 * @receiver The left operand in the multiplication.
 * @param right The right operand in the multiplication.
 *
 * @return The result of the multiplication operation as a `Mul` object.
 */
context(Ops, MathOps)
operator fun <T> Operand<T>.times(right: Operand<T>): Mul<T> = mul(this, right)

/**
 * Defines the division operation for two operands in the context of TensorFlow operations and
 * math operations.
 *
 * @param divisor The operand to be divided by.
 * @return An operand which represents the result of the division.
 */
context(Ops, MathOps)
operator fun <T> Operand<T>.div(divisor: Operand<T>): Div<T> = div(this, divisor)

/**
 * Overloads the division operator for a tensor and a float.
 *
 * This extension function provides a more readable syntax for dividing a TensorFlow operand
 * (tensor) by a float.
 * It wraps around the `div` operation provided by TensorFlow and returns the result as a new
 * tensor.
 *
 * @receiver The tensor operand in the division.
 * @param divisor The float value that divides the tensor.
 *
 * @return The result of the division operation as a `Div` object.
 */
context(Ops, MathOps)
operator fun Operand<Float>.div(divisor: Float): Div<Float> = div(this, constant(divisor))

/**
 * Defines the addition operation for a Float scalar and an operand in the context of TensorFlow
 * operations and math operations.
 *
 * @param right The operand to be added to the Float scalar.
 * @return An operand which represents the result of the addition.
 */
context(Ops, MathOps)
operator fun Float.plus(right: Operand<Float>): Operand<Float> =
    add(constant(this), right)

/**
 * Defines the negation operation for an operand in the context of TensorFlow operations and math
 * operations.
 *
 * @return An operand which represents the result of the negation.
 */
context(Ops, MathOps)
operator fun <T> Operand<T>.unaryMinus(): Operand<T> = neg(this)

/**
 * Returns the maximum of a float and a tensor.
 *
 * This function provides a way to compute the element-wise maximum of a float and a TensorFlow
 * operand (tensor).
 * It wraps around the `maximum` operation provided by TensorFlow.
 *
 * @param a The float to be compared.
 * @param b The tensor to be compared.
 *
 * @return A tensor with the same shape as `b`, where each element is the maximum between the float
 * `a` and the corresponding element in `b`.
 */
context(Ops, MathOps)
fun maximum(a: Float, b: Operand<Float>): Operand<Float> = maximum(constant(a), b)

/**
 * Returns the minimum of a float and a tensor.
 *
 * This function provides a way to compute the element-wise minimum of a float and a TensorFlow
 * operand (tensor).
 * It wraps around the `minimum` operation provided by TensorFlow.
 *
 * @param a The float to be compared.
 * @param b The tensor to be compared.
 *
 * @return A tensor with the same shape as `b`, where each element is the minimum between the float
 * `a` and the corresponding element in `b`.
 */
context(Ops, MathOps)
fun minimum(a: Float, b: Operand<Float>): Operand<Float> = minimum(constant(a), b)


/**
 * Converts a Tensor of a generic type into a Float tensor.
 *
 * This function uses the [Tensor.floatValue] method for conversion.
 *
 * @return A Float value representing the converted Tensor.
 */
fun <T> Tensor<T>.toFloat() = floatValue()
