/**
 * "skorch" (c) by Ignacio Slater M.
 * "skorch" is licensed under a
 * Creative Commons Attribution 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by/4.0/>.
 */
package cl.ravenhill.skorch

import kotlin.properties.Delegates

object THGenerator {
  private var left by Delegates.notNull<Int>()
  private var seeded by Delegates.notNull<Int>()
  private var normalIsValid by Delegates.notNull<Int>()

  fun createUnseeded(): THGenerator {
    left = 1
    seeded = 0
    normalIsValid = 0
    return this
  }

  fun create(): THGenerator {

  }

  fun SkorchOps.rand(x: Int, y: Int): Int {
    return 1
  }