import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    kotlin("jvm") version "1.9.0"
    application
}

group = "cl.ravenhill"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-api:0.6.0-alpha-1")
    implementation("org.jetbrains.kotlinx:kotlin-deeplearning-tensorflow:0.6.0-alpha-1")
}

tasks.test {
    useJUnitPlatform()
}

application {
    mainClass.set("MainKt")
}

tasks.withType<KotlinCompile> {
    compilerOptions {
        freeCompilerArgs.add("-Xcontext-receivers")
    }
}