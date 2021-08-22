import cl.ravenhill.skorch.rand
import cl.ravenhill.skorch.SkorchOps as skorch

fun main(args: Array<String>) {
  println("Hello World!")
  val x = skorch.rand(5, 3)
  println(x)
}