#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

uint8_t* openImage(const char* path, int& width, int& height, int& channels) {

    uint8_t* pixelData = stbi_load(path, &width, &height, &channels, 0);

    if (!pixelData) {
        std::cout << "Could not upen image " << path << std::endl;
        std::cout << stbi_failure_reason() << std::endl;
    }

    return pixelData;
}

#pragma pack(push, 1)
// DDS file header structure
struct DDSHeader {
    uint32_t size;              // Header size (must be 124)
    uint32_t flags;             // Flags to indicate valid fields
    uint32_t height;            // Height of the texture
    uint32_t width;             // Width of the texture
    uint32_t pitchOrLinearSize; // Pitch or linear size
    uint32_t depth;             // Depth of the texture (3D textures)
    uint32_t mipMapCount;       // Number of mipmaps
    uint32_t reserved[11];      // Reserved fields
    struct {
        uint32_t size;          // Size of this structure (32 bytes)
        uint32_t flags;         // Flags to indicate valid fields
        uint32_t fourCC;        // FourCC code for compressed formats
        uint32_t rgbBitCount;   // Number of bits per pixel (unused for DXT1)
        uint32_t rBitMask;      // Red bit mask (unused for DXT1)
        uint32_t gBitMask;      // Green bit mask (unused for DXT1)
        uint32_t bBitMask;      // Blue bit mask (unused for DXT1)
        uint32_t aBitMask;      // Alpha bit mask (unused for DXT1)
    } pixelFormat;
    uint32_t caps[4];           // Capabilities
    uint32_t reserved2;         // Reserved
};
#pragma pack(pop)

//write DXT1-compressed data into a DDS file
bool WriteDDS_DXT1(const fs::path& path, const std::vector<uint8_t>& compressedData, uint32_t width, uint32_t height, uint32_t mipMapCount = 1) {
    // Open file for binary writing
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << path << "\n";
        return false;
    }

    // Write DDS magic number
    const uint32_t DDS_MAGIC = 0x20534444; // "DDS "
    file.write(reinterpret_cast<const char*>(&DDS_MAGIC), sizeof(DDS_MAGIC));

    // Calculate linear size of the top-level mipmap
    uint32_t blockSize = 8; // DXT1 compression uses 8 bytes per 4x4 block
    uint32_t linearSize = ((width + 3) / 4) * ((height + 3) / 4) * blockSize;

    // Create DDS header
    DDSHeader header = {};
    header.size = 124;
    header.flags = 0x00021007; // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_LINEARSIZE
    header.height = height;
    header.width = width;
    header.pitchOrLinearSize = linearSize;
    header.mipMapCount = mipMapCount;
    header.pixelFormat.size = 32;
    header.pixelFormat.flags = 0x4; // DDPF_FOURCC
    header.pixelFormat.fourCC = 0x31545844; // "DXT1" in little-endian
    header.pixelFormat.rgbBitCount = 0; // Not used for FOURCC
    header.pixelFormat.rBitMask = 0;
    header.pixelFormat.gBitMask = 0;
    header.pixelFormat.bBitMask = 0;
    header.pixelFormat.aBitMask = 0;
    header.caps[0] = 0x1000; // DDSCAPS_TEXTURE
    if (mipMapCount > 1) {
        header.caps[0] |= 0x400000; // DDSCAPS_MIPMAP
        header.caps[1] = 0x8;      // DDSCAPS_COMPLEX
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // Write compressed data
    file.write(reinterpret_cast<const char*>(compressedData.data()), compressedData.size());

    // Close file
    file.close();

    return true;
}