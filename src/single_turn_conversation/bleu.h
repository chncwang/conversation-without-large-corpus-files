#ifndef SINGLE_TURN_CONVERSATION_BLEU_H
#define SINGLE_TURN_CONVERSATION_BLEU_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <boost/format.hpp>
#include "conversation_structure.h"
#include "print.h"
#include "tinyutf8.h"

using namespace std;

struct CandidateAndReferences {
    vector<string> candidate;
    vector<vector<string>> references;

    CandidateAndReferences() = default;

    CandidateAndReferences(const vector<string> &c, const vector<vector<string>> &ref) {
        candidate = c;
        references = ref;
    }
};

class PunctuationSet {
public:
    set<string> punctuation_set;

    PunctuationSet() {
        for (int i = 0; i < PUNCTUATIONS.length(); ++i) {
            utf8_string punc = PUNCTUATIONS.substr(i, 1);
            punctuation_set.insert(punc.cpp_str());
            cout << "PunctuationSet - punc:" << punc.cpp_str() << endl;
        }
    }

private:
    static const utf8_string PUNCTUATIONS;
};
const utf8_string PunctuationSet::PUNCTUATIONS =
        "~`!@#$%^&*()_-+={[}]|:;\"'<>,.?/，。！『』；：？、（）「」《》“”";

bool includePunctuation(const string &str) {
    static PunctuationSet set;
    utf8_string utf8_str = str;
    for (int i = 0; i < utf8_str.length(); ++i) {
        if (set.punctuation_set.find(utf8_str.substr(i, 1).cpp_str()) !=
                set.punctuation_set.end()) {
            return true;
        }
    }
    return false;
}

float mostMatchedCount(const CandidateAndReferences &candidate_and_references,
        int gram_len) {
    using namespace std;

    int max_mached_count = 0;
    const auto &references = candidate_and_references.references;
    const auto &candidate = candidate_and_references.candidate;
    if (candidate.size() < gram_len) {
        return 0;
    }
    vector<string> matched_ref;
    for (const vector<string> &reference : references) {
        if (reference.size() < gram_len) {
            continue;
        }
        int matched_count = 0;
        vector<bool> matched;
        for (int j = 0; j < reference.size() + 1 - gram_len; ++j) {
            matched.push_back(false);
        }
        for (int i = 0; i < candidate.size() + 1 - gram_len; ++i) {
            for (int j = 0; j < reference.size() + 1 - gram_len; ++j) {
                if (matched.at(j)) {
                    continue;
                }

                bool finded = false;
                for (int k = 0; k < gram_len; ++k) {
                    if (candidate.at(i + k) != reference.at(j + k)) {
                        break;
                    }
                    if (k == gram_len - 1) {
                        finded = true;
                    }
                }

                if (finded) {
                    matched.at(j) = true;
                    matched_count++;
                    break;
                }
            }
        }

        if (matched_count > max_mached_count) {
            max_mached_count = matched_count;
            matched_ref = reference;
        }
    }
//    if (max_mached_count > 0) {
//        cout << "candidate:" << endl;
//        print(candidate);
//        cout << "max_mached_count:" << max_mached_count << " gram len:" << gram_len << endl;
//        print(matched_ref);
//    }

    return max_mached_count;
}

int mostMatchedLength(const CandidateAndReferences &candidate_and_references) {
    int candidate_len = candidate_and_references.candidate.size();
    auto cmp = [&](const vector<string> &a, const vector<string> &b)->bool {
        int dis_a = candidate_len - a.size();
        int dis_b = candidate_len - b.size();
        return abs(dis_a) < abs(dis_b);
    };
    const auto &e = min_element(candidate_and_references.references.begin(),
            candidate_and_references.references.end(), cmp);
//    cout << "candidate:" << endl;
//    print(candidate_and_references.candidate);
//    cout << "most match len:" << e->size() << endl;
//    print(*e);
    return e->size();
}

float computeBleu(const vector<CandidateAndReferences> &candidate_and_references_vector,
        int max_gram_len) {
    using namespace std;
    float weighted_sum = 0.0f;
    int r_sum = 0;
    int c_sum = 0;

    for (int i = 1; i <=max_gram_len; ++i) {
        int matched_count_sum = 0;
        int candidate_count_sum = 0;
        int candidate_len_sum = 0;
        for (const auto &candidate_and_references : candidate_and_references_vector) {
            int matched_count = mostMatchedCount(candidate_and_references, i);
            matched_count_sum += matched_count;
            candidate_count_sum += candidate_and_references.candidate.size() + 1 - i;
            candidate_len_sum += candidate_and_references.candidate.size();

            int r = mostMatchedLength(candidate_and_references);
            r_sum += r;
        }
        c_sum += candidate_len_sum;

        weighted_sum += 1.0f / max_gram_len * log(static_cast<float>(matched_count_sum) /
                candidate_count_sum);
        cout << boost::format("matched_count:%1% candidate_count:%2% weighted_sum%3%") %
            matched_count_sum % candidate_count_sum % weighted_sum << endl;
    }

    float bp = c_sum > r_sum ? 1.0f : exp(1 - static_cast<float>(r_sum) / c_sum);
    cout << boost::format("candidate sum:%1% ref:%2% bp:%3%") % c_sum % r_sum % bp << endl;
    return bp * exp(weighted_sum);
}

#endif
