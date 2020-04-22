#ifndef SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H
#define SINGLE_TURN_CONVERSATION_SRC_BASIC_DATA_MANAGER_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <codecvt>
#include <fstream>
#include <iterator>
#include <regex>
#include <iostream>
#include <utility>
#include <atomic>
#include <mutex>
#include "single_turn_conversation/conversation_structure.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/default_config.h"
#include "tinyutf8.h"
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/asio.hpp>

using namespace std;
using boost::format;
using namespace boost::asio;

std::vector<PostAndResponses> readPostAndResponsesVector(const std::string &filename) {
    std::vector<PostAndResponses> results;
    std::string line;
    std::ifstream ifs(filename);
    while (std::getline(ifs, line)) {
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(":"));
        if (strs.size() != 2) {
            abort();
        }
        int post_id = stoi(strs.at(0));
        PostAndResponses post_and_responses;
        post_and_responses.post_id = post_id;
        std::vector<std::string> strs2;
        boost::split(strs2, strs.at(1), boost::is_any_of(","));
        if (strs2.empty()) {
            cerr << "readPostAndResponsesVector - no response id found!" << line << endl;
            abort();
        }
        for (std::string &str : strs2) {
            post_and_responses.response_ids.push_back(stoi(str));
        }
        results.push_back(std::move(post_and_responses));
    }

    return results;
}

std::vector<ConversationPair> toConversationPairs(const PostAndResponses &post_and_responses) {
    std::vector<ConversationPair> results;
    for (int response_id : post_and_responses.response_ids) {
        ConversationPair conversation_pair(post_and_responses.post_id, response_id);
        results.push_back(std::move(conversation_pair));
    }
    return results;
}

std::vector<ConversationPair> toConversationPairs(
        const std::vector<PostAndResponses> &post_and_responses_vector) {
    std::vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        std::vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (const ConversationPair & conversation_pair : conversation_pairs) {
            results.push_back(conversation_pair);
        }
    }
    return results;
}

std::vector<ConversationPair> readConversationPairs(const std::string &filename) {
    std::vector<PostAndResponses> post_and_responses_vector = readPostAndResponsesVector(filename);
    std::vector<ConversationPair> results;
    for (const PostAndResponses &post_and_responses : post_and_responses_vector) {
        std::vector<ConversationPair> conversation_pairs = toConversationPairs(post_and_responses);
        for (ConversationPair &conversation_pair : conversation_pairs) {
            results.push_back(std::move(conversation_pair));
        }
    }

    return results;
}

bool isPureChinese(const string &word) {
    std::regex expression("^[\u4e00-\u9fff]+$");
    return std::regex_search(word, expression);
}

bool containChinese(const utf8_string &word) {
    return word.size() != word.length();
}

bool isPureEnglish(const utf8_string &word) {
    if (containChinese(word)) {
        return false;
    }
    for (int i = 0; i < word.length(); ++i) {
        char c = word.at(i);
        if (!((c == '-' || c == '.' || c == '/' || c == ':' || c == '_') || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
            return false;
        }
    }
    return true;
}

bool isPureNumber(const utf8_string &word) {
    for (int i = 0; i < word.length(); ++i) {
        char c = word.at(i);
        if (!(c == '.' || (c >= '0' && c <= '9'))) {
            return false;
        }
    }
    return true;
}

std::vector<std::vector<std::string>> readSentences(const std::string &filename) {
    std::string line;
    std::ifstream ifs(filename);
    std::vector<std::vector<std::string>> results;

    int i = 0;
    while (std::getline(ifs, line)) {
        std::vector<std::string> strs;
        boost::split_regex(strs, line, boost::regex("##"));
        int index = stoi(strs.at(0));
        if (i != index) {
            abort();
        }

        const std::string &sentence = strs.at(1);
        std::vector<string> words;
        boost::split(words, sentence, boost::is_any_of(" "));
        std::vector<utf8_string> utf8_words;
        for (const string &word : words) {
            utf8_string s(word);
            utf8_words.push_back(s);
        }

        std::vector<std::string> characters;
        for (const utf8_string &word : utf8_words) {
            if (isPureEnglish(word) && !isPureNumber(word)) {
                string w;
                for (int i = 0; i < word.length(); ++i) {
                    char c = word.at(i);
                    if (c >= 'A' && c <= 'Z') {
                        c += 'a' - 'A';
                    }
                    w += c;
                }
                characters.push_back(w);
            } else {
                characters.push_back(word.cpp_str());
            }
//            if (isPureEnglish(word) && !isPureNumber(word)) {
//                string w;
//                for (int i = 0; i < word.length(); ++i) {
//                    char c = word.at(i);
//                    if (c >= 'A' && c <= 'Z') {
//                        c += 'a' - 'A';
//                    }
//                    w += c;
//                }
//                characters.push_back(w);
//            } else {
//                for (int i = 0; i < word.length(); ++i) {
//                    characters.push_back(word.substr(i, 1).cpp_str());
//                }
//            }
        }

        characters.push_back(STOP_SYMBOL);
        results.push_back(characters);
        if (i % 10000 == 0) {
            cout << boost::format("i:%1%\n") % i;
            for (const string &c : characters) {
                cout << c << " ";
            }
            cout << endl;
        }
        ++i;
    }

    return results;
}

vector<string> reprocessSentence(const vector<string> &sentence,
        const unordered_map<string, int> &word_counts,
        int min_occurences) {
    vector<string> processed_sentence;
    for (const string &word : sentence) {
        if (isPureChinese(word)) {
            auto it = word_counts.find(word);
            int occurence;
            if (it == word_counts.end()) {
                cout << format("word not found:%1%\n") % word;
                occurence = 0;
            } else {
                occurence = it->second;
            }
            if (occurence <= min_occurences) {
                for (int i = 0; i < word.size(); i += 3) {
                    processed_sentence.push_back(word.substr(i, 3));
                }
            } else {
                processed_sentence.push_back(word);
            }
        } else {
            processed_sentence.push_back(word);
        }
    }
    return processed_sentence;
}

vector<vector<string>> reprocessSentences(const vector<vector<string>> &sentences,
        const unordered_set<string> &words,
        const unordered_set<int> &ids) {
    cout << boost::format("sentences size:%1%") % sentences.size() << endl;

    thread_pool pool(16);
    vector<vector<string>> result;
    map<int, vector<string>> result_map;
    mutex result_mutex;
    mutex cout_mutex;
    atomic_int i(0);
    int id = 0;
    for (const auto &sentence : sentences) {
        auto f = [&, id]() {
            if (i % 1000 == 0) {
                cout_mutex.lock();
                cout << static_cast<float>(i) / sentences.size() << endl;
                cout_mutex.unlock();
            }
            vector<string> processed_sentence;
            if (ids.find(id) == ids.end()) {
                processed_sentence = sentence;
            } else {
                for (const string &word : sentence) {
                    if (isPureChinese(word)) {
                        auto it = words.find(word);
                        if (it == words.end()) {
                            for (int i = 0; i < word.size(); i += 3) {
                                processed_sentence.push_back(word.substr(i, 3));
                            }
                        } else {
                            processed_sentence.push_back(word);
                        }
                    } else {
                        processed_sentence.push_back(word);
                    }
                }
            }
            result_mutex.lock();
            result_map.insert(make_pair(id, processed_sentence));
            result_mutex.unlock();
            ++i;
        };
        post(pool, f);
        ++id;
    }
    pool.join();

    for (int i = 0; i < sentences.size(); ++i) {
        auto it = result_map.find(i);
        if (it == result_map.end()) {
            cerr << boost::format("id %1% not found\n") % i;
            abort();
        }
        result.push_back(it->second);
    }

    return result;
}

void reprocessSentences(const vector<PostAndResponses> bundles,
        vector<vector<string>> &posts,
        vector<vector<string>> &responses,
        const unordered_map<string, int> &word_counts,
        int min_occurences) {
    vector<vector<string>> result;

    cout << "bund count:" << bundles.size() << endl;
    int i = 0;

    for (const PostAndResponses &bundle : bundles) {
        cout << i++ / (float)bundles.size() << endl;
        int post_id = bundle.post_id;
        auto &post = posts.at(post_id);
        post = reprocessSentence(post, word_counts, min_occurences);
        for (int response_id : bundle.response_ids) {
            auto &res = responses.at(response_id);
            res = reprocessSentence(res, word_counts, min_occurences);
        }
    }
}

std::vector<std::string> readBlackList(const std::string &filename) {
    std::string line;
    std::ifstream ifs(filename);
    std::vector<std::string> result;
    cout << "black:" << endl;
    while (std::getline(ifs, line)) {
        cout << line << endl;
        result.push_back(line);
    }
    return result;
}

vector<unordered_set<string>> toWordSets(const vector<vector<string>> &sentences) {
    vector<unordered_set<string>> results;
    for (const auto & v : sentences) {
        unordered_set<string> word_set;
        for (const auto & w : v) {
            word_set.insert(w);
        }
        results.push_back(move(word_set));
    }

    return results;
}

unordered_map<string, int> toOccurenceMap(const vector<unordered_set<string>> &word_sets) {
    unordered_map<string, int> word_frequencies;
    for (const auto &word_set : word_sets) {
        for (const string &w : word_set) {
            const auto &it = word_frequencies.find(w);
            if (it == word_frequencies.end()) {
                word_frequencies.insert(make_pair(w, 1));
            } else {
                it->second++;
            }
        }
    }
    return word_frequencies;
}

unordered_map<string, float> toIdfMap(const unordered_map<string, int> &word_frequencies,
        int size) {
    unordered_map<string, float> result;
    for (const auto &it : word_frequencies) {
        float idf = log((float)size / it.second);
        result.insert(make_pair(it.first, idf));
    }
    return result;
}

unordered_map<string, unordered_map<string, float>> calPMI(
        const vector<vector<string>> &post_sentences,
        const vector<vector<string>> &response_sentences,
        const vector<ConversationPair> &pairs) {
    cout << "calculating post word sets..." << endl;
    vector<unordered_set<string>> post_word_sets = toWordSets(post_sentences);
    cout << "calculating response word sets..." << endl;
    vector<unordered_set<string>> response_word_sets = toWordSets(response_sentences);
    cout << "calculating post occurence map..." << endl;
    unordered_map<string, int> occurence_map = toOccurenceMap(post_word_sets);
    cout << "calculating response occurence map..." << endl;
    unordered_map<string, int> response_occurence_map = toOccurenceMap(response_word_sets);
    cout << "calculating post idf map..." << endl;
    unordered_map<string, float> post_idf_map = toIdfMap(occurence_map, post_word_sets.size());
    unordered_map<string, unordered_map<string, int>> conditional_post_occurence_map;
    for (const ConversationPair &conv_pair : pairs) {
        const unordered_set<string> &post_word_set = post_word_sets.at(conv_pair.post_id);
        for (const string &post_word : post_word_set) {
            const auto &outer_it = conditional_post_occurence_map.find(post_word);
            unordered_map<string, int> *p;
            if (outer_it == conditional_post_occurence_map.end()) {
                unordered_map<string, int> m;
                conditional_post_occurence_map.insert(make_pair(post_word, move(m)));
                p = &conditional_post_occurence_map.find(post_word)->second;
            } else {
                p = &outer_it->second;
            }
            const unordered_set<string> &response_word_set =
                response_word_sets.at(conv_pair.response_id);
            for (const string &response_word : response_word_set) {
                const auto &it = p->find(response_word);
                if (it == p->end()) {
                    p->insert(make_pair(response_word, 1));
                } else {
                    it->second++;
                }
            }
        }
    }

    unordered_map<string, unordered_map<string, float>> pmi_map;
    for (const auto &outter_it : conditional_post_occurence_map) {
        unordered_map<string, float> m;
        for (const auto &inner_it : outter_it.second) {
            float pmi;
            if (outter_it.first == STOP_SYMBOL || inner_it.first == STOP_SYMBOL) {
                pmi = 0;
            } else {
                pmi = log((float)inner_it.second / response_occurence_map.at(inner_it.first)) +
                    post_idf_map.at(outter_it.first);
//                if (pmi > 0)
//                    cout << "post:" << outter_it.first << " response:" << inner_it.first << pmi << endl;
            }
            m.insert(make_pair(inner_it.first, pmi));
        }
        pmi_map.insert(make_pair(outter_it.first, move(m)));
    }

    return pmi_map;
}

string getMostRelatedKeyword(const vector<string> &post,
        const unordered_map<string, unordered_map<string, float>> &pmi_map) {
    unordered_map<string, float> scores;
    for (const string &post_word : post) {
        const auto &it = pmi_map.find(post_word);
        if (it == pmi_map.end()) {
            cout << "warning: post word " << post_word << " not found" << endl;
        } else {
            for (const auto &inner_it : it->second) {
                auto scores_it = scores.find(inner_it.first);
                if (scores_it == scores.end()) {
                    scores.insert(make_pair(inner_it.first, 0.0f));
                    scores_it = scores.find(inner_it.first);
                }
                scores_it->second += inner_it.second;
            }
        }
    }
    float max_score = -1;
    const string *most_related_keyword = nullptr;
    for (const auto &it : scores) {
        if (it.second > max_score) {
            max_score = it.second;
            most_related_keyword = &it.first;
        }
    }
    if (most_related_keyword == nullptr) {
        cerr << "getMostRelatedKeyword - most_related_keyword is nullptr" << endl;
        abort();
    }
    return *most_related_keyword;
}

string getKeyword(const vector<string> &post, const vector<string> &response,
        const unordered_map<string, unordered_map<string, float>> &pmi_map) {
    vector<float> word_pmis;
    for (const auto &w : response) {
        float sum = 0.0f;
        for (const auto &post_w : post) {
            try {
                sum += pmi_map.at(post_w).at(w);
            } catch (const exception &e) {
                sum -= 1e-10;
            }
        }
        word_pmis.push_back(sum / post.size());
    }

    float max_v = -1;
    string word;
    for (int j = 0; j < word_pmis.size(); ++j) {
        if (word_pmis.at(j) >= max_v) {
            word = response.at(j);
            max_v = word_pmis.at(j);
        }
    }

    return word;
}

#endif
