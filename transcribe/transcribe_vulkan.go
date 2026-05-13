//go:build vulkan
// +build vulkan

package transcribe

/*
#cgo LDFLAGS: ${SRCDIR}/../libs/libggml-vulkan.a -lvulkan
*/
import "C"
