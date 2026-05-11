BINARY := dictation-go

.PHONY: all clean

all:
	go build -o $(BINARY) .

clean:
	rm -f $(BINARY)
