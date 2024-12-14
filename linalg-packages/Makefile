# Устанавливаем компилятор C++ (g++)
CXX = g++

# Флаги компилятора:
# -std=c++11                               : используем стандарт C++11
# -O3                                      : оптимизация уровня 3
# -Wall                                    : включаем все предупреждения
# $(shell python3 -m pybind11 --includes)  : добавляем пути включения для pybind11
CXXFLAGS = -std=c++11 -O3 -Wall $(shell python3 -m pybind11 --includes)

# Флаги компоновщика для создания разделяемой библиотеки Python:
# $(shell python3-config --ldflags)        : получаем флаги компоновки для Python
# -shared                                  : создаем разделяемую библиотеку
# -fPIC                                    : генерируем позиционно-независимый код
PY_LDFLAGS = $(shell python3-config --ldflags) -shared -fPIC

# Путь к директории с исходным кодом C++
SRC_DIR = src

# Путь к директории с Python-обертками
PYTHON_DIR = python

# Цель по умолчанию: собирает цель 'vector-entropy'
all: vector-entropy

# Компонуем объектные файлы в модуль расширения для Python
# $^                                     : все зависимости (bindings.o и entropy.o)
# -o                                     : указываем имя выходного файла
# $(PYTHON_DIR)/entropy_core$(shell python3-config --extension-suffix) : имя выходного файла с соответствующим расширением для Python
# $(PY_LDFLAGS)                          : флаги компоновщика для Python
vector-entropy: $(PYTHON_DIR)/bindings.o $(SRC_DIR)/entropy.o
	$(CXX) $^ -o $(PYTHON_DIR)/entropy_core$(shell python3-config --extension-suffix) $(PY_LDFLAGS)

# Компилируем файл bindings.cpp в объектный файл
# $(CXX)         : используем компилятор C++
# $(CXXFLAGS)    : флаги компилятора
# -fPIC          : генерируем позиционно-независимый код
# -c             : компилируем без компоновки
# $<             : первый зависимый файл (bindings.cpp)
# -o $@          : имя выходного файла (bindings.o)
$(PYTHON_DIR)/bindings.o: $(PYTHON_DIR)/bindings.cpp $(SRC_DIR)/entropy.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Компилируем файл entropy.cpp в объектный файл
# $(CXX)         : используем компилятор C++
# $(CXXFLAGS)    : флаги компилятора
# -fPIC          : генерируем позиционно-независимый код
# -c             : компилируем без компоновки
# $<             : первый зависимый файл (entropy.cpp)
# -o $@          : имя выходного файла (entropy.o)
$(SRC_DIR)/entropy.o: $(SRC_DIR)/entropy.cpp $(SRC_DIR)/entropy.h
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# Цель очистки: удаляет все объектные файлы и скомпилированный модуль расширения
# rm -f                                    : удаляет файлы без подтверждения
# $(PYTHON_DIR)/*.o                        : удаляет все объектные файлы в директории python
# $(SRC_DIR)/*.o                           : удаляет все объектные файлы в директории src
# $(PYTHON_DIR)/entropy_core$(shell python3-config --extension-suffix) : удаляет скомпилированный модуль расширения для Python
clean:
	rm -f $(PYTHON_DIR)/*.o $(SRC_DIR)/*.o $(PYTHON_DIR)/entropy_core$(shell python3-config --extension-suffix)
