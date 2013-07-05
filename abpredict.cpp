/*
Copyright (c) 2013, Koichiro Yamaguchi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include "readSampleDataFile.h"
#include "AdaBoost.h"

struct ParameterABPredict {
    bool verbose;
    std::string testDataFilename;
    std::string modelFilename;
    bool outputScoreFile;
    std::string outputScorelFilename;
};

// Prototype declaration
void exitWithUsage();
ParameterABPredict parseCommandline(int argc, char* argv[]);

void exitWithUsage() {
    std::cerr << "usage: abtrain [options] test_set_file model_file" << std::endl;
    std::cerr << "options:" << std::endl;
    std::cerr << "   -o: output score file" << std::endl;
    std::cerr << "   -v: verbose" << std::endl;
    
    exit(1);
}

ParameterABPredict parseCommandline(int argc, char* argv[]) {
    ParameterABPredict parameters;
    parameters.verbose = false;
    parameters.outputScoreFile = false;
    parameters.outputScorelFilename = "";
    
    // Options
    int argIndex;
    for (argIndex = 1; argIndex < argc; ++argIndex) {
        if (argv[argIndex][0] != '-') break;
        
        switch (argv[argIndex][1]) {
            case 'v':
                parameters.verbose = true;
                break;
            case 'o':
            {
                ++argIndex;
                if (argIndex >= argc) exitWithUsage();
                parameters.outputScoreFile = true;
                parameters.outputScorelFilename = argv[argIndex];
                break;
            }
            default:
                std::cerr << "error: undefined option" << std::endl;
                exitWithUsage();
                break;
        }
    }
    
    // Test data file
    if (argIndex >= argc) exitWithUsage();
    parameters.testDataFilename = argv[argIndex];
    
    // Model file
    ++argIndex;
    if (argIndex >= argc) exitWithUsage();
    parameters.modelFilename = argv[argIndex];
    
    return parameters;
}

int main(int argc, char* argv[]) {
    ParameterABPredict parameters = parseCommandline(argc, argv);
    
    if (parameters.verbose) {
        std::cerr << std::endl;
        std::cerr << "Test data: " << parameters.testDataFilename << std::endl;
        std::cerr << "Model:     " << parameters.modelFilename << std::endl;
        if (parameters.outputScoreFile) {
            std::cerr << "Output score: " << parameters.outputScorelFilename << std::endl;
        }
        std::cerr << std::endl;
    }

    AdaBoost adaBoost;
    adaBoost.readFile(parameters.modelFilename);
    
    std::vector< std::vector<double> > testSamples;
    std::vector<bool> testLabels;
    readSampleDataFile(parameters.testDataFilename, testSamples, testLabels);
    int testSampleTotal = static_cast<int>(testSamples.size());

    std::ofstream outputScoreStream;
    if (parameters.outputScoreFile) {
        outputScoreStream.open(parameters.outputScorelFilename.c_str(), std::ios_base::out);
        if (outputScoreStream.fail()) {
            std::cerr << "error: can't open file (" << parameters.outputScorelFilename << ")" << std::endl;
            exit(1);
        }
    }

    int positiveTotal = 0;
    int positiveCorrectTotal = 0;
    int negativeTotal = 0;
    int negativeCorrectTotal = 0;
    for (int sampleIndex = 0; sampleIndex < testSampleTotal; ++sampleIndex) {
        double score = adaBoost.predict(testSamples[sampleIndex]);
        
        if (testLabels[sampleIndex]) {
            ++positiveTotal;
            if (score > 0) ++positiveCorrectTotal;
        } else {
            ++negativeTotal;
            if (score <= 0) ++negativeCorrectTotal;
        }
        
        if (parameters.outputScoreFile) {
            outputScoreStream << score << std::endl;
        }
    }
    if (parameters.outputScoreFile) {
        outputScoreStream.close();
    }


    double accuracyAll = static_cast<double>(positiveCorrectTotal + negativeCorrectTotal)/(positiveTotal + negativeTotal);
    std::cout << "Accuracy = " << accuracyAll;
    std::cout << " (" << positiveCorrectTotal + negativeCorrectTotal << " / " << positiveTotal + negativeTotal << ")" << std::endl;
    std::cout << "  positive: " << static_cast<double>(positiveCorrectTotal)/positiveTotal;
    std::cout << " (" << positiveCorrectTotal << " / " << positiveTotal << "), ";
    std::cout << "negative: " << static_cast<double>(negativeCorrectTotal)/negativeTotal;
    std::cout << " (" << negativeCorrectTotal << " / " << negativeTotal << ")" << std::endl;
}
