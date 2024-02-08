.PHONY: all clean debug release test lib

all: release

build/Makefile:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release

release: build/Makefile
	$(MAKE) -C build
	@echo "Release build complete. Executables are located in the build/ directory."

debug: build/Makefile
	$(MAKE) -C build CMAKE_BUILD_TYPE=Debug
	@echo "Debug build complete. Executables are located in the build/ directory."

lib: build/Makefile
	$(MAKE) -C build mlp
	@echo "Library build complete. Library is located in the build/ directory."

iris_train: release
	./build/iris_train

iris_predict: release
	./build/iris_predict

test: release
	$(MAKE) -C build test
	@echo "Running tests complete."

clean:
	rm -rf build/