#include <iostream>
#include <string>
#include <fstream>
#include <tuple>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

int main()
{
    const string input = "../prediction.csv";
    const string output = "../submission.csv";

    ifstream ifile{input};

    vector<pair<string, double>> data;
    string line;
    ifile >> line; // skip the header
    while (ifile >> line) {
        const auto pos = line.find(',');
        auto key = line.substr(0, pos);
        const auto value = std::stof(line.substr(pos + 1));
        data.emplace_back(move(key), value);
    }
    ifile.close();
    cout << "Pairs read: " << data.size() << endl;

    std::sort(data.begin(), data.end(),
              [](const auto& x, const auto& y) {
                  return x.first < y.first;
              });

    cout << "Sorted" << endl;

    vector<pair<string, double>> result;
    string key = data[0].first;
    double sum = 0;
    for (const auto& pair : data) {
        if (key != pair.first) {
            result.emplace_back(key, log(sum + 1));
            sum = 0;
        }
        sum += exp(pair.second) - 1;
        key = pair.first;
    }
    result.emplace_back(key, log(sum + 1));

    cout << "Combined" << endl;

    ofstream ofile{output};
    ofile << "fullVisitorId,PredictedLogRevenue" << endl;
    for (const auto& pair : result) {
        ofile << pair.first << ',' << pair.second << endl;
    }

    cout << "Pairs written: " << result.size() << endl;
}
