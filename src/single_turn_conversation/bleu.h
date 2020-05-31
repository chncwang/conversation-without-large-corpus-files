#ifndef SINGLE_TURN_CONVERSATION_BLEU_H
#define SINGLE_TURN_CONVERSATION_BLEU_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "print.h"
#include <boost/format.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include "conversation_structure.h"
#include "print.h"
#include "tinyutf8.h"
#include "mteval/NISTEvaluator.h"
#include "mteval/BLEUEvaluator.h"
#include "N3LDG.h"

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
        int gram_len,
        bool print_log = false) {
    using namespace std;

    int max_mached_count = 0;
    string max_matched_log;
    const auto &references = candidate_and_references.references;
    const auto &candidate = candidate_and_references.candidate;
    if (candidate.size() < gram_len) {
        return 0;
    }
    vector<string> matched_ref;
    for (const vector<string> &reference : references) {
        string log;
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
                    log += (boost::format("%1%gram match:") % gram_len).str();
                    for (int k = 0; k < gram_len; ++k) {
                        log += candidate.at(i + k) + " ";
                    }
                    log += "\n";
                    break;
                }
            }
        }

        if (matched_count > max_mached_count) {
            max_mached_count = matched_count;
            matched_ref = reference;
            max_matched_log = log;
        }
    }

    if (max_mached_count > 0 && print_log) {
        cout << "candidate:" << endl;
        print(candidate);
        cout << max_matched_log;
        cout << "max_mached_count:" << max_mached_count << " gram len:" << gram_len << endl;
        print(matched_ref);
    }

    return max_mached_count;
}

int puncRemovedLen(const vector<string> &sentence) {
    return sentence.size();
//    int len = 0;
//    for (const string &w : sentence) {
//        if (!includePunctuation(w)) {
//            ++len;
//        }
//    }
//    return len;
}

int mostMatchedLength(const CandidateAndReferences &candidate_and_references) {
    int candidate_len = puncRemovedLen(candidate_and_references.candidate);
    auto cmp = [&](const vector<string> &a, const vector<string> &b)->bool {
        int a_len = puncRemovedLen(a);
        int dis_a = candidate_len - a_len;
        int b_len = puncRemovedLen(b);
        int dis_b = candidate_len - b_len;
        return abs(dis_a) < abs(dis_b);
    };
    const auto &e = min_element(candidate_and_references.references.begin(),
            candidate_and_references.references.end(), cmp);
//    cout << "candidate len:" << candidate_len << endl;
//    print(candidate_and_references.candidate);
//    cout << "most match len:" << e->size() << endl;
//    for (const auto &e : candidate_and_references.references) {
//        cout << "other:" << e.size() << " ";
//    }
//    cout << endl;
//    print(*e);
    return puncRemovedLen(*e);
}

float ngramCount(const vector<string> sentence, int ngram) {
    int len = sentence.size() + 1 - ngram;
    return len;
}

vector<string> toChars(const vector<string> &src) {
    vector<string> result;
    for (const string &w : src) {
        utf8_string utf8(w);
        for (int i = 0; i < utf8.length(); ++i) {
            result.push_back(utf8.substr(i, 1).cpp_str());
        }
    }
    return result;
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
        int j = 0;
        for (const auto &candidate_and_references : candidate_and_references_vector) {
            int matched_count = mostMatchedCount(candidate_and_references, i,
                    ++j == candidate_and_references_vector.size() && 4 == max_gram_len);
            matched_count_sum += matched_count;
            candidate_count_sum += ngramCount(candidate_and_references.candidate, i);
            candidate_len_sum += puncRemovedLen(candidate_and_references.candidate);

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

float computeMtevalBleu(const vector<CandidateAndReferences> &candidate_and_references_vector,
        int max_gram_len) {
    using namespace MTEval;
    EvaluatorParam param;
    param.name = "ngram";
    param.int_val = max_gram_len;
    BLEUEvaluator evaluator({param});
    vector<Sample> samples;
    for (const CandidateAndReferences &e : candidate_and_references_vector) {
        Sample sample;
        sample.hypothesis = e.candidate;
        sample.references = e.references;
        evaluator.prepare(sample);
        samples.push_back(move(sample));
    }

    Statistics stats;
    for (const Sample sample : samples) {
        stats += evaluator.map(sample);
    }

    float score = evaluator.integrate(stats);

    return score;
}

float computeMtevalBleu(const CandidateAndReferences &candidate_and_references,
        int max_gram_len) {
    vector<CandidateAndReferences> v = {candidate_and_references};
    return computeMtevalBleu(v, max_gram_len);
}

void computeMtevalBleuForEachResponse(
        const vector<CandidateAndReferences> &candidate_and_references_vector,
        int max_gram_len,
        float &avg,
        float &standard_deviation) {
    using namespace boost::accumulators;

    vector<float> bleus;
    for (const auto &e : candidate_and_references_vector) {
        float bleu = computeMtevalBleu(e, max_gram_len);
        bleus.push_back(bleu);
    }

    avg = accumulate(bleus.begin(), bleus.end(), 0.0f) / bleus.size();
    vector<float> zero_centralized_squares;
    for (float bleu : bleus) {
        float v = bleu - avg;
        zero_centralized_squares.push_back(v * v);
    }
    float variance = zero_centralized_squares.size() == 1 ? 0 :
        accumulate(zero_centralized_squares.begin(), zero_centralized_squares.end(), 0.0f) /
        (zero_centralized_squares.size() - 1);
    standard_deviation = sqrt(variance);
}

float computeNist(const vector<CandidateAndReferences> &candidate_and_references_vector,
        int max_gram_len) {
    using namespace MTEval;
    EvaluatorParam param;
    param.name = "ngram";
    param.int_val = max_gram_len;
    NISTEvaluator evaluator({param});
    vector<Sample> samples;
    for (const CandidateAndReferences &e : candidate_and_references_vector) {
        Sample sample;
        sample.hypothesis = e.candidate;
        sample.references = e.references;
        evaluator.prepare(sample);
        samples.push_back(move(sample));
    }

    Statistics stats;
    for (const Sample sample : samples) {
        stats += evaluator.map(sample);
    }

    float score = evaluator.integrate(stats);

    return score;
}

float computeEntropy(const vector<CandidateAndReferences> &candidate_and_references_vector,
        const unordered_map<string, float> &idf_table) {
    float idf_sum = 0;
    int len_sum = 0;
    for (const CandidateAndReferences &e : candidate_and_references_vector) {
        const auto &s = e.candidate;
        for (const string &word : s) {
             const auto &it = idf_table.find(word);
             if (it == idf_table.end()) {
                 cerr << "word " << word << " not found" << endl;
                 abort();
             }
             float idf = it->second;
             idf_sum += idf;
        }
        len_sum += s.size();
    }
    return idf_sum / len_sum;
}

float computeMatchedEntropy(const vector<CandidateAndReferences> &candidate_and_references_vector,
        const unordered_map<string, float> &idf_table) {
    float idf_sum = 0;
    int len_sum = 0;
    for (const CandidateAndReferences &e : candidate_and_references_vector) {
        const auto &s = e.candidate;
        float max_idf = -1;
        for (const auto &ref : e.references) {
            vector<bool> used;
            used.resize(ref.size());
            for (int i = 0; i < used.size(); ++i) {
                used.at(i) = false;
            }
            float idf_inner_sum = 0;
            for (const string &word : s) {
                const auto &it = idf_table.find(word);
                if (it == idf_table.end()) {
                    cerr << "word " << word << " not found" << endl;
                    abort();
                }
                float idf = it->second;
                int i = 0;
                for (const auto &ref_word : ref) {
                    if (!used.at(i) && ref_word == it->first) {
                        used.at(i) = true;
                        idf_inner_sum += idf;
                    }
                    ++i;
                }
            }
            if (idf_inner_sum > max_idf) {
                max_idf = idf_inner_sum;
            }
        }
        idf_sum += max_idf;
        len_sum += s.size();
    }
    return idf_sum / len_sum;
}

float computeDist(const vector<CandidateAndReferences> &candidate_and_references_vector,
        int ngram) {
    unordered_set<string> distinctions;
    int sentence_len_sum = 0;
    for (const auto &e : candidate_and_references_vector) {
        const auto &s = e.candidate;
        int sentence_size = s.size();
        int len = sentence_size - ngram + 1;
        sentence_len_sum += std::max<int>(len, 0);
        if (s.size() >= ngram) {
            for (int begin_i = 0; begin_i < sentence_size - ngram + 1; ++begin_i) {
                string ngram_str;
                for (int pos_i = begin_i; pos_i < ngram; ++pos_i) {
                    ngram_str += s.at(pos_i);
                }
                distinctions.insert(ngram_str);
            }
        }
    }
    return static_cast<float>(distinctions.size()) / sentence_len_sum;
}

float vectorCos(const dtype *a, const dtype *b, int len) {
    float inner_prod_sum = 0;
    float a_len_square = 0;
    float b_len_square = 0;

    for (int i = 0; i < len; ++i) {
        inner_prod_sum += a[i] * b[i];
        a_len_square += a[i] * a[i];
        b_len_square += b[i] * b[i];
    }

    return inner_prod_sum / sqrt(a_len_square) / sqrt(b_len_square);
}

float greedyMatching(const vector<string> &a, const vector<string> &b,
        LookupTable<Param>& embedding_table) {
    float max_cos_sum = 0;
    for (const auto &candidate_word : a) {
        float max_cos = -2;
        for (const auto &ref_word : b) {
            int candidate_id = embedding_table.elems.from_string(candidate_word);
            dtype *candidate_vector = embedding_table.E.val[candidate_id];
            int ref_id = embedding_table.elems.from_string(ref_word);
            dtype *ref_vector = embedding_table.E.val[ref_id];
            float cos = vectorCos(candidate_vector, ref_vector, embedding_table.E.outDim());
            if (cos > max_cos) {
                max_cos = cos;
            }
        }
        max_cos_sum += max_cos;
    }
    return max_cos_sum / a.size();
}

float computeGreedyMatching(const CandidateAndReferences &candidate_and_refs,
        LookupTable<Param>& embedding_table) {
    const auto &refs = candidate_and_refs.references;
    float max_g = -2;
    for (const auto &ref : refs) {
        auto known_ref = ref;
        for (auto &w : known_ref) {
            if (!embedding_table.findElemId(w)) {
                w = unknownkey;
            }
        }
        float g = 0.5 * (greedyMatching(known_ref, candidate_and_refs.candidate, embedding_table) +
            greedyMatching(candidate_and_refs.candidate, known_ref, embedding_table));
        if (g > max_g) {
            max_g = g;
        }
    }
    return max_g;
}

const string NGRAM_SEG = "*%*";

set<string> ngramKeys(const vector<string> &sentence, int ngram) {
    set<string> results;
    if (sentence.size() >= ngram) {
        for (int i = 0; i < sentence.size() + 1 - ngram; ++i) {
            string key;
            for (int j = 0; j < ngram; ++j) {
                key += sentence.at(i + j) + NGRAM_SEG;
            }
            results.insert(key);
        }
    }
    return results;
}

unordered_map<string, float> computeNgramIdf(const vector<vector<vector<string>>> &sentences,
        int ngram) {
    unordered_map<string, int> count_table;
    for (const auto pair : sentences) {
        set<string> pair_set;
        for (const auto sentence : pair) {
            if (sentence.size() < ngram) {
                continue;
            }
            for (int i = 0; i < sentence.size() + 1 - ngram; ++i) {
                string key;
                for (int j = 0; j < ngram; ++j) {
                    key += sentence.at(i + j) + NGRAM_SEG;
                }
                pair_set.insert(key);
            }
        }

        for (const string &w : pair_set) {
            const auto &it = count_table.find(w);
            if (it == count_table.end()) {
                count_table.insert(make_pair(w, 1));
            } else {
                count_table.at(w)++;
            }
        }
    }

    unordered_map<string, float> idf_table;
    for (const auto &it : count_table) {
        float idf = log(sentences.size() / static_cast<float>(it.second));
        idf_table.insert(make_pair(it.first, idf));
    }
    return idf_table;
}

vector<float> zeroVector(int size) {
    vector<float> result;
    result.resize(size);
    for (float &e : result) {
        e = 0;
    }
    return result;
}

float occurenceIdf(const vector<string> &sentence, int ngram, const string &word, float idf) {
    if (sentence.size() < ngram) {
        return 0;
    }

    float result = 0;

    for (int i = 0; i < sentence.size() + 1 - ngram; ++i) {
        string key;
        for (int j = 0; j < ngram; ++j) {
            key += sentence.at(i + j) + NGRAM_SEG;
        }
        if (key == word) {
            result += idf;
        }
    }
    return result;
}

float vectorLen(const vector<float> &v) {
    float square_sum = 0;
    for (float n : v) {
        square_sum += n * n;
    }
    return sqrt(square_sum);
}

float vectorCos(const vector<float> &a, const vector<float> &b) {
    if (a.size() != b.size()) {
        cerr << "vectorCos - input sizes are not equal" << endl;
        abort();
    }

    float sum = 0;
    for (int i = 0; i < a.size(); ++i) {
        sum += a.at(i) * b.at(i);
    }
    int a_len = vectorLen(a);
    int b_len = vectorLen(b);
    if (a_len == 0 || b_len == 0) {
        return 0;
    }
    return sum / (a_len * b_len);
}

float computeCIDEr(const CandidateAndReferences &candidate_and_references,
        const unordered_map<string, float> idf_table,
        int ngram) {
    set<string> can_words = ngramKeys(candidate_and_references.candidate, ngram);
    float sum = 0;
    for (const auto &ref : candidate_and_references.references) {
        set<string> words = ngramKeys(ref, ngram);
        for (const string &cw : can_words) {
            words.insert(cw);
        }
        auto can_vec = zeroVector(words.size());
        auto ref_vec = zeroVector(words.size());
        int offset = 0;
        for (const string &word : words) {
            const auto &it = idf_table.find(word);
            float idf = it == idf_table.end() ? 0 : it->second;
            can_vec.at(offset) = occurenceIdf(candidate_and_references.candidate, ngram, word,
                    idf);
            ref_vec.at(offset) = occurenceIdf(ref, ngram, word, idf);
            offset++;
        }
        float cos = vectorCos(can_vec, ref_vec);
        cout << "can:";
        print(candidate_and_references.candidate);
        cout << "ref:";
        print(ref);
        cout << "cos:" << cos << endl << endl;
        sum += cos;
    }
    return sum / candidate_and_references.references.size();
}


#endif
