#include <string>
#include <vector>

void readSampleDataFile(const std::string sampleDataFilename,
                        std::vector< std::vector<double> >& sampleFeatures,
                        std::vector<bool>& sampleLabels);
