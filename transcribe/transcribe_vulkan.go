//go:build vulkan
// +build vulkan

package transcribe

/*
#cgo LDFLAGS: ${SRCDIR}/../libs/libggml-vulkan.a -lm
#cgo !windows LDFLAGS: -lstdc++
#cgo windows LDFLAGS: -Wl,-Bstatic -Wl,--start-group -lstdc++ -lwinpthread -Wl,--end-group -Wl,-Bdynamic
*/
import "C"
