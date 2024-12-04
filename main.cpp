#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <cassert>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <array>
#include <cmath>
#include <bitset>

namespace fs = std::filesystem;

uint8_t* openImage(const fs::path& path, int& width, int& height, int& channels) {

	uint8_t* pixelData = stbi_load(path.string().c_str(), &width, &height, &channels, 0);

	if (!pixelData) {
		std::cout << "Could not upen image" << std::endl;
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

// Function to write DXT1-compressed data into a DDS file
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

uint8_t quantizeToNBits(uint8_t input, int N) {

    uint8_t quantizedValue = static_cast<uint8_t>(std::round(input * (std::pow(2, N) - 1) / 255.0));

    return quantizedValue;
}

int main(int, char**) {

	int width, height, channels;
	uint8_t* data = openImage(INPUT_DIR "/image.jpeg", width, height, channels);

	std::cout << "width: " << width << std::endl;
	std::cout << "height: " << height << std::endl;
	std::cout << "channels: " << channels << std::endl;
	std::cout << "size: " << width * height * channels << std::endl;

    int paddedWidth = (width + 3) / 4;
    int paddedHeight = (height + 3) / 4;

    std::vector<uint8_t> compressedData(paddedWidth * paddedHeight * 8); // DXT1 compressed data size

    int abba = 0;

    for (int i = 0; i < paddedHeight; i++) {
        for (int j = 0; j < paddedWidth; j++) {

            //c0_lo, c0_hi, c1_lo, c1_hi, bits_0, bits_1, bits_2, bits_3

            uint8_t R, G, B;

            if (i * 4 >= height || j * 4 >= width) {
                R = 0x00;
                G = 0x00;
                B = 0x00;
            }
            else {
                R = data[(i * 4 * width + j * 4) * channels + 0];
                G = data[(i * 4 * width + j * 4) * channels + 1];
                B = data[(i * 4 * width + j * 4) * channels + 2];
            }

            uint16_t R16 = quantizeToNBits(R, 5);
            uint16_t G16 = quantizeToNBits(G, 6);
            uint16_t B16 = quantizeToNBits(B, 5);

            uint16_t color0 = (R16 << 11) + (G16 << 5) + B16;

            uint16_t color1 = 0;

            int blockIndex = (i * paddedWidth + j) * 8;

            compressedData[blockIndex + 0] = color0 & 0xFF;;
            compressedData[blockIndex + 1] = (color0 >> 8) & 0xFF;
            compressedData[blockIndex + 2] = color1 & 0xFF;;
            compressedData[blockIndex + 3] = (color1 >> 8) & 0xFF;
            compressedData[blockIndex + 4] = 0x00;
            compressedData[blockIndex + 5] = 0x00;
            compressedData[blockIndex + 6] = 0x00;
            compressedData[blockIndex + 7] = 0x00;

            /*if (abba < 5) {
                //std::cout << i * width * 4 * 3 + j * 4 * 3 << " " << i * width * 4 * 3 + j * 4 * 3 + 1 << " " << i * width * 4 * 3 + j * 4 * 3 + 2 << std::endl;

                std::cout << "R: " << std::bitset<16>(R16) << std::endl;
                std::cout << "G: " << std::bitset<16>(G16) << std::endl;
                std::cout << "B: " << std::bitset<16>(B16) << std::endl;

                std::cout << "color0: " << std::bitset<16>(color0) << std::endl;

                abba++;
            }*/

        }
    }


    if (WriteDDS_DXT1(OUTPUT_DIR "./output_dxt1.dds", compressedData, width, height)) {
        std::cout << "DDS file with DXT1 compression written successfully.\n";
    }
    else {
        std::cout << "Failed to write DDS file.\n";
    }


	return 0;
}