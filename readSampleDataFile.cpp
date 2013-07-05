#include "readSampleDataFile.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct FeatureElement {
    int index;
    double value;
};

bool readLine(FILE* inputFile, char*& lineBuffer, int& maxLineLength);

void readSampleDataFile(const std::string sampleDataFilename,
                        std::vector< std::vector<double> >& sampleFeatures,
                        std::vector<bool>& sampleLabels)
{
    FILE* dataFile;
    dataFile = fopen(sampleDataFilename.c_str(), "r");
    if (dataFile == NULL) {
        std::cerr << "error: can't open file (" << sampleDataFilename << ")" << std::endl;
        exit(1);
    }
    
    int maxLineLength = 1024;
    char* lineBuffer = reinterpret_cast<char*>(malloc(maxLineLength));
    
    std::vector< std::vector<FeatureElement> > featureElementList;
    std::vector<int> labelList;
    
    int featureDimension = 0;
    
    while (readLine(dataFile, lineBuffer, maxLineLength)) {
        char* labelChar = strtok(lineBuffer, " \t");
        char* endPointer;
        int label = static_cast<int>(strtol(labelChar, &endPointer, 10));
        if (endPointer == labelChar) {
            std::cerr << "error: bad format in data file (" << sampleDataFilename << ")" << std::endl;
            exit(1);
        }
        labelList.push_back(label);
        
        std::vector<FeatureElement> featureElements;
        while (1) {
            char* indexChar = strtok(NULL, ":");
            char* valueChar = strtok(NULL, " \t");
            if (valueChar == NULL) break;
            
            FeatureElement newElement;
            newElement.index = static_cast<int>(strtol(indexChar, &endPointer, 10));
            if (endPointer == indexChar || *endPointer != '\0' || newElement.index <= 0) {
                std::cerr << "error: bad format in data file (" << sampleDataFilename << ")" << std::endl;
                exit(1);
            }
            newElement.value = strtod(valueChar, &endPointer);
            if (endPointer == valueChar || (*endPointer != '\0' && !isspace(*endPointer))) {
                std::cerr << "error: bad format in data file (" << sampleDataFilename << ")" << std::endl;
                exit(1);
            }
            featureElements.push_back(newElement);
            
            if (newElement.index > featureDimension) featureDimension = newElement.index;
        }
        featureElementList.push_back(featureElements);
    }
    fclose(dataFile);
    free(lineBuffer);
    
    int sampleTotal = static_cast<int>(featureElementList.size());
    
    sampleFeatures.resize(sampleTotal);
    sampleLabels.resize(sampleTotal);
    for (int sampleIndex = 0; sampleIndex < sampleTotal; ++sampleIndex) {
        if (labelList[sampleIndex] > 0) sampleLabels[sampleIndex] = true;
        else sampleLabels[sampleIndex] = false;

        sampleFeatures[sampleIndex].resize(featureDimension);
        for (int i = 0; i < featureDimension; ++i) sampleFeatures[sampleIndex][i] = 0;
        for (int elementIndex = 0; elementIndex < static_cast<int>(featureElementList[sampleIndex].size()); ++elementIndex) {
            sampleFeatures[sampleIndex][featureElementList[sampleIndex][elementIndex].index - 1]
                = featureElementList[sampleIndex][elementIndex].value;
        }
    }
}


bool readLine(FILE* inputFile, char*& lineBuffer, int& maxLineLength) {
    if (fgets(lineBuffer, maxLineLength, inputFile) == NULL) return false;
    
    while (strrchr(lineBuffer, '\n') == NULL) {
        maxLineLength *= 2;
        lineBuffer = reinterpret_cast<char*>(realloc(lineBuffer, maxLineLength));
        int readLength = static_cast<int>(strlen(lineBuffer));
        if (fgets(lineBuffer+readLength, maxLineLength-readLength, inputFile) == NULL) break;
    }
    
    return true;
}
