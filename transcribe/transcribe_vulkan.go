//go:build vulkan

package transcribe

/*
#cgo LDFLAGS: ${SRCDIR}/../libs/libggml-vulkan.a -lm
#cgo linux LDFLAGS: -Wl,-Bstatic -lstdc++ -Wl,-Bdynamic
#cgo darwin LDFLAGS: -lstdc++
#cgo windows LDFLAGS: -Wl,-Bstatic -Wl,--start-group -lstdc++ -lwinpthread -Wl,--end-group -Wl,-Bdynamic
*/
import "C"
