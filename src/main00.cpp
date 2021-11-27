//
// Created by Valerii on 18/11/2021.
//
//#ifndef __main00_cpp
//#define __main00_cpp
#pragma once

#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>

#define CL_DEVICE_BUILT_IN_KERNELS 0x103F

namespace lab00 {
    template <typename T>
    std::string to_string(T value)
    {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }

    void reportError(cl_int err, const std::string &filename, int line)
    {
        if (CL_SUCCESS == err)
            return;

        // Таблица с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
        throw std::runtime_error(message);
    }

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

    std::vector<cl_device_id> getDeviceIds(cl_platform_id platform) {
        cl_uint size = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &size));

        std::vector<cl_device_id> result(size);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, size, result.data(), nullptr));
        return result;
    }
    std::vector<unsigned char> getPlatformInfo(cl_platform_id platform, cl_platform_info paramName) {
        size_t size = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, paramName, 0, nullptr, &size));

        std::vector<unsigned char> result(size, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, paramName, size, result.data(), nullptr));
        return result;
    }

    std::vector<unsigned char> getDeviceInfo(cl_device_id  device, cl_device_info  param_name) {
        size_t size = 0;
        OCL_SAFE_CALL(clGetDeviceInfo (device, param_name, 0, nullptr, &size));
        std::vector<unsigned char> result(size, 0);
        OCL_SAFE_CALL(clGetDeviceInfo (device, param_name, size, result.data(), nullptr));
        return result;
    }

    std::vector<cl_platform_id> getPlatformIDs() {
        cl_uint platformsCount = 0;
        OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));

        // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
        std::vector<cl_platform_id> platforms(platformsCount);
        OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
        return platforms;
    }

    cl_device_type getDeviceType(cl_device_id pDeviceId) {
        cl_device_type deviceType;
        OCL_SAFE_CALL(clGetDeviceInfo (pDeviceId, CL_DEVICE_TYPE, sizeof deviceType, &deviceType, nullptr));
        return deviceType;
    }

    void printDeviceInfo(cl_device_id pDeviceId);

    int main()
    {
        // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
        if (!ocl_init())
            throw std::runtime_error("Can't init OpenCL driver!");

        // Откройте
        // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
        // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
        // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
        std::vector<cl_platform_id> platforms = getPlatformIDs();
        cl_uint platformsCount = platforms.size();
        std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

        for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
            std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
            cl_platform_id platform = platforms[platformIndex];

            // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
            // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
//        size_t platformNameSize = 0;
//        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
            // TODO 1.1
            // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
            // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
            // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
            // Откройте таблицу с кодами ошибок:
            // libs/clew/CL/cl.h:103
            // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
            // Найдите там нужный код ошибки и ее название
            // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
            // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
            // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

            // TODO 1.2
            // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
//        std::vector<unsigned char> platformName(platformNameSize, 0);
            // clGetPlatformInfo(...);
//        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
//        std::cout << "    Platform name: " << platformName.data() << std::endl;
            std::cout << "    Platform name: " << getPlatformInfo(platform, CL_PLATFORM_NAME).data() << std::endl;

            // TODO 1.3
            // Запросите и напечатайте так же в консоль вендора данной платформы
            std::cout << "    Platform vendor: " << getPlatformInfo(platform, CL_PLATFORM_VENDOR).data() << std::endl;

            // TODO 2.1
            // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
            const std::vector<cl_device_id> deviceIds = getDeviceIds(platform);
            cl_uint devicesCount = deviceIds.size();

            for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
                // TODO 2.2
                // Запросите и напечатайте в консоль:
                cl_device_id pDeviceId = deviceIds[deviceIndex];
                // - Название устройства
                printDeviceInfo(pDeviceId);


            }
        }

        return 0;
    }

    void printDeviceInfo(cl_device_id pDeviceId) {
        std::cout << "      Device name: " << getDeviceInfo(pDeviceId, CL_DEVICE_NAME).data() << std::endl;
        // - Тип устройства (видеокарта/процессор/что-то странное)
//                cl_device_type deviceType;
//                OCL_SAFE_CALL(clGetDeviceInfo (pDeviceId, CL_DEVICE_TYPE, sizeof deviceType, &deviceType, nullptr));
        std::cout << "      Device type: " << getDeviceType(pDeviceId) << std::endl;

        // - Размер памяти устройства в мегабайтах
        cl_ulong globalMemSizeBytes;
        OCL_SAFE_CALL(
                clGetDeviceInfo(pDeviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof globalMemSizeBytes, &globalMemSizeBytes,
                                nullptr));
        std::cout << "      Global Mem(Bytes): " << globalMemSizeBytes << std::endl;
        std::cout << "      Global Mem(MB): " << ( globalMemSizeBytes / (1024 * 1024)) << std::endl;

        // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
        cl_ulong localMemSizeBytes;
        OCL_SAFE_CALL(clGetDeviceInfo(pDeviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof localMemSizeBytes, &localMemSizeBytes,
                                      nullptr));
        std::cout << "      Local Mem(Bytes): " << localMemSizeBytes << std::endl;


        std::cout << "      Device extensions: " << getDeviceInfo(pDeviceId, CL_DEVICE_EXTENSIONS).data() << std::endl;
        std::cout << "      Device OpenCl version: " << getDeviceInfo(pDeviceId, CL_DEVICE_OPENCL_C_VERSION).data() << std::endl;
        std::cout << "      Device profile: " << getDeviceInfo(pDeviceId, CL_DEVICE_PROFILE).data() << std::endl;
        std::cout << "      Device version: " << getDeviceInfo(pDeviceId, CL_DEVICE_VERSION).data() << std::endl;
        std::cout << "      Device driver version: " << getDeviceInfo(pDeviceId, CL_DRIVER_VERSION).data() << std::endl;
        std::cout << "      Device built in kernels: " << getDeviceInfo(pDeviceId, CL_DEVICE_BUILT_IN_KERNELS).data() << std::endl;
    }
}

//#endif  /* __main00_cpp */