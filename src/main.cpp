#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>
#include "main00.cpp"


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

cl_command_queue createCommandQueue(cl_context pContext, const std::vector<cl_device_id>& devices);

cl_mem createROBuffer(cl_context pContext, std::vector<float>& data);

cl_mem createWOBuffer(cl_context pContext, std::vector<float>& data);

cl_program createProgram(cl_context pContext, std::string basicString);

std::vector<unsigned char> getBuildLog(cl_program pProgram, const std::vector<cl_device_id> deviceIds);

cl_kernel createKernel(cl_program pProgram, const char *kernelName);

template<typename T>
size_t sizeofVector(const typename std::vector<T>& vec);


std::vector<cl_device_id> findDevices() {
    std::vector<cl_device_id> gpus;
    std::vector<cl_device_id> cpus;
    auto platforms = lab00::getPlatformIDs();
    for (auto &platformId: platforms) {
        auto deviceIds = lab00::getDeviceIds(platformId);
        for (auto &deviceId: deviceIds) {
            auto deviceType = lab00::getDeviceType(deviceId);
            if ((deviceType & CL_DEVICE_TYPE_GPU) != 0) {
                gpus.push_back(deviceId);
            }
            if ((deviceType & CL_DEVICE_TYPE_CPU) != 0) {
                cpus.push_back(deviceId);
            }
        }
    }
    return !gpus.empty() ? gpus : cpus;
}

cl_context createContext(const std::vector<cl_device_id>& deviceIds) {
    cl_context_properties properties[] = {0};
    cl_int err = 0;
    cl_context pContext = clCreateContext(properties, deviceIds.size(), deviceIds.data(), nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);
    return pContext;
}

int main() {
//    lab00::main();
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init()) {
        throw std::runtime_error("Can't init OpenCL driver!");
    }



    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    const std::vector<cl_device_id> deviceIds = findDevices();
    std::cout << "Devices count: " << deviceIds.size() << "!" << std::endl;

    for (const auto &deviceId: deviceIds) {
        lab00::printDeviceInfo(deviceId);
    }

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_context pContext = createContext(deviceIds);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    cl_command_queue pCommandQueue = createCommandQueue(pContext, deviceIds);

    unsigned int n = 1000 * 1000;// * 500; // todo  * 1000
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    cl_mem as_gpu = createROBuffer(pContext, as);
    cl_mem bs_gpu = createROBuffer(pContext, bs);
    cl_mem cs_gpu = createWOBuffer(pContext, cs);


    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    cl_program program = createProgram(pContext, kernel_sources);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    cl_int buildProgramStatusCode = clBuildProgram(program, deviceIds.size(), deviceIds.data(), nullptr, nullptr, nullptr);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    auto buildLog = getBuildLog(program, deviceIds);
//    size_t log_size = 0;
//    std::vector<char> log(log_size, 0);
    if (buildLog.size() > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << buildLog.data() << std::endl;
    }

    OCL_SAFE_CALL(buildProgramStatusCode);

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects


    auto kernel = createKernel(program, "aplusb");

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &as_gpu);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &bs_gpu);
        clSetKernelArg(kernel, i++, sizeof(cl_mem), &cs_gpu);
        clSetKernelArg(kernel, i++, sizeof (unsigned int), &n);
        // clSetKernelArg(kernel, i++, ..., ...);
        // clSetKernelArg(kernel, i++, ..., ...);
        // clSetKernelArg(kernel, i++, ..., ...);
        // clSetKernelArg(kernel, i++, ..., ...);
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полученного события - см. в документации подходящий метод среди Event Objects
    int bytesInGb = 1024 * 1024 * 1024;
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        timer t1;
        int laps = 20;
        for (unsigned int i = 0; i < laps; ++i) {
            // clEnqueueNDRangeKernel...
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(pCommandQueue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, &event));
            // clWaitForEvents...
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        double elapsed = t1.elapsed();
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "Kernel elapsed: " << elapsed << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        long double totalKernels = (long double) laps * n;
        long double gFlops = totalKernels / (1000 * 1000 * 1000 * elapsed);
        std::cout << "GFlops: " << gFlops << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        long double bandwidth = totalKernels * 3 * sizeof(float) / (bytesInGb * elapsed);
        std::cout << "VRAM bandwidth: " << bandwidth << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        timer t1;
        size_t bufferSizeBytes = sizeofVector(cs);
        int laps = 20;
        for (unsigned int i = 0; i < laps; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(pCommandQueue, cs_gpu, CL_TRUE, 0, bufferSizeBytes, cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        double elapsed = t1.elapsed();
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "Result data transfer time(total elapsed): " << elapsed << " s" << std::endl;
        long double bandwidth = ((long double) laps * bufferSizeBytes) / (elapsed * bytesInGb);
        std::cout << "VRAM -> RAM bandwidth: " << bandwidth << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            auto message = "i: " + to_string(i) + ", cs[i]: " + to_string(cs[i]) + ", as[i] + bs[i]: " + to_string(as[i] + bs[i]);
            throw std::runtime_error("CPU and GPU results differ! " + message);
        }
    }

    OCL_SAFE_CALL(clReleaseMemObject(as_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(bs_gpu));
    OCL_SAFE_CALL(clReleaseMemObject(cs_gpu));
    OCL_SAFE_CALL(clReleaseCommandQueue(pCommandQueue));
    OCL_SAFE_CALL(clReleaseContext(pContext));

    return 0;
}

cl_kernel createKernel(cl_program pProgram, const char *kernelName) {
    cl_int err = 0;
    cl_kernel pKernel = clCreateKernel(pProgram, kernelName, &err);
    OCL_SAFE_CALL(err);
    return pKernel;
}

std::vector<unsigned char> getBuildLog(cl_program pProgram, const std::vector<cl_device_id> deviceIds) {
    size_t size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(pProgram, deviceIds[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &size));
    std::vector<unsigned char> result(size);
    OCL_SAFE_CALL(clGetProgramBuildInfo(pProgram, deviceIds[0], CL_PROGRAM_BUILD_LOG, result.size(), result.data(), nullptr));
    return result;
}

cl_program createProgram(cl_context pContext, std::string basicString) {
    cl_int err = 0;
    const char *tmpPString = basicString.c_str();
    unsigned long long int tmpStringSize = basicString.size();
    cl_program pProgram = clCreateProgramWithSource(pContext, 1, &tmpPString, &tmpStringSize, &err);
    OCL_SAFE_CALL(err);
    return pProgram;
}

cl_mem createBuffer(cl_context pContext, std::vector<float> &data, cl_mem_flags memFlags) {
    cl_int err = 0;
    float *hostPointer = (memFlags & CL_MEM_COPY_HOST_PTR) ? data.data() : nullptr;
    cl_mem result = clCreateBuffer(pContext, memFlags, sizeofVector(data), hostPointer, &err);
    OCL_SAFE_CALL(err);
    return result;
}

cl_mem createWOBuffer(cl_context pContext, std::vector<float> &data) {
    return createBuffer(pContext, data, CL_MEM_WRITE_ONLY);
}

cl_mem createROBuffer(cl_context pContext, std::vector<float> &data) {
    return createBuffer(pContext, data, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
}

cl_command_queue createCommandQueue(cl_context pContext, const std::vector<cl_device_id>& devices) {
    cl_int err = 0;
    cl_command_queue result = clCreateCommandQueue(pContext, devices[0], 0, &err);
    OCL_SAFE_CALL(err);
    return result;
}

template<typename T>
size_t sizeofVector(const std::vector<T> &vec) {
    return sizeof(T) * vec.size();
}
