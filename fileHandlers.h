#ifndef IMAGE_DDS_WRITER_H
#define IMAGE_DDS_WRITER_H

#include <vector>
#include <cstdint>
#include <filesystem>

namespace fs = std::filesystem;

// Function to load an image using stb_image
uint8_t* openImage(const char* path, int& width, int& height, int& channels);

// Structure for DDS file header
#pragma pack(push, 1)
struct DDSHeader {
    uint32_t size;
    uint32_t flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitchOrLinearSize;
    uint32_t depth;
    uint32_t mipMapCount;
    uint32_t reserved[11];
    struct {
        uint32_t size;
        uint32_t flags;
        uint32_t fourCC;
        uint32_t rgbBitCount;
        uint32_t rBitMask;
        uint32_t gBitMask;
        uint32_t bBitMask;
        uint32_t aBitMask;
    } pixelFormat;
    uint32_t caps[4];
    uint32_t reserved2;
};
#pragma pack(pop)

// Function to write DXT1-compressed data to a DDS file
bool WriteDDS_DXT1(const fs::path& path, const std::vector<uint8_t>& compressedData, uint32_t width, uint32_t height, uint32_t mipMapCount = 1);

#endif // IMAGE_DDS_WRITER_H