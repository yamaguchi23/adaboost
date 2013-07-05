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

#ifndef ADABOOST_H
#define ADABOOST_H

#include <string>
#include <vector>

class AdaBoost {
public:
    AdaBoost(const int boostingType = 2)
        : boostingType_(boostingType), featureTotal_(0), sampleTotal_(0) {}
    
    void setBoostingType(const int boostingType);
    void setTrainingSamples(const std::string& trainingDataFilename);
    
    void train(const int roundTotal, const bool verbose = false);
    
    double predict(const std::vector<double>& featureVector) const;
    
    void writeFile(const std::string filename) const;
    void readFile(const std::string filename);
    
private:
    class DecisionStump {
    public:
        DecisionStump() : featureIndex_(-1), error_(-1) {}
        
        void set(const int featureIndex,
                 const double threshold,
                 const double outputLarger,
                 const double outputSmaller,
                 const double error = -1);
        
        double evaluate(const double featureValue) const;
        double evaluate(const std::vector<double>& featureVector) const;
        
        int featureIndex() const { return featureIndex_; }
        double threshold() const { return threshold_; }
        double outputLarger() const { return outputLarger_; }
        double outputSmaller() const { return outputSmaller_; }
        double error() const { return error_; }
        
    private:
        int featureIndex_;
        double threshold_;
        double outputLarger_;
        double outputSmaller_;
        double error_;
    };
    
    void initializeWeights();
    void sortSampleIndices();
    void trainRound();
    void calcWeightSum();
    DecisionStump learnOptimalClassifier(const int featureIndex);
    void computeClassifierOutputs(const double weightSumLarger,
                                  const double weightLabelSumLarger,
                                  const double positiveWeightSumLarger,
                                  const double negativeWeightSumLarger,
                                  double& outputLarger,
                                  double& outputSmaller) const;
    double computeError(const double positiveWeightSumLarger,
                        const double negativeWeightSumLarger,
                        const double outputLarger,
                        const double outputSmaller) const;
    void updateWeight(const DecisionStump& bestClassifier);

    int boostingType_;
    int featureTotal_;
    std::vector<DecisionStump> weakClassifiers_;
    
    // Training samples
    int sampleTotal_;
    std::vector< std::vector<double> > samples_;
    std::vector<bool> labels_;
    std::vector<double> weights_;

    // Data for training
    std::vector< std::vector<int> > sortedSampleIndices_;
    double weightSum_;
    double weightLabelSum_;
    double positiveWeightSum_;
    double negativeWeightSum_;
};

#endif
