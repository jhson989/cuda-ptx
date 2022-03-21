CC = nvcc
PROGRAM = program.out
SRCS = main.cu src/cuda_c.cu src/cuda_ptx.cu
INCS = include/cuda_c.cuh include/cuda_ptx.cuh

.PHONY : all run clean

all: ${PROGRAM}

${PROGRAM}: ${SRCS} ${INC} Makefile
	${CC} -o $@ ${SRCS}


run : ${PROGRAM}
	./${PROGRAM}

clean :
	rm ${PROGRAM}


