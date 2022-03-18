CC = nvcc
PROGRAM = program.out
SRCS = main.cu
INCS = 

.PHONY : clean all

all: ${PROGRAM}

${PROGRAM}: ${SRCS} ${INC} Makefile
	${CC} -o $@ ${SRCS}

clean :
	rm ${PROGRAM}


