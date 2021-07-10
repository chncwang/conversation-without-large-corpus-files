#ifndef SINGLE_TURN_CONVERSATION_PERPLEX_H
#define SINGLE_TURN_CONVERSATION_PERPLEX_H

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <boost/format.hpp>

#include "insnet/insnet.h"

using namespace std;
using namespace insnet;

float computePerplex(const std::vector<Node *> &nodes, const std::vector<int> &answers, int len,
        int &hit_count_result,
        vector<int> &keyword_hit_flags,
        vector<int> &token_hit_flags,
        vector<int> &unified_token_hit_flags,
        float &keyword_ppl,
        float &normal_ppl) {
    keyword_ppl = 0;
    normal_ppl = 0;
    float log_sum = 0.0f;
    keyword_hit_flags.clear();
    token_hit_flags.clear();
    unified_token_hit_flags.clear();
    vector<int> hit_flags;
    int hit_beam = 5;

    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
#if USE_GPU
        const_cast<insnet::Node &>(node).val().copyFromDeviceToHost();
#endif
        dtype reciprocal_answer_prob = 1 / node.getVal().v[answers.at(i)];
        dtype l = log(reciprocal_answer_prob);
        log_sum += l;
        bool hit = true;
        int hit_count = 0;
        for (int j = 0; j < node.size(); ++j) {
            if (node.getVal().v[j] >= node.getVal().v[answers.at(i)]) {
                if (++hit_count > hit_beam) {
                    hit = false;
                    break;
                }
            }
        }
        hit_flags.push_back(hit);
        if (i < len) {
            token_hit_flags.push_back(hit);
            normal_ppl += l;
        } else {
            keyword_ppl += l;
        }
    }

    int hit_count = 0;
    int j_begin = 0;

    for (int i = len; i < nodes.size(); ++i) {
        bool keyword_hit = hit_flags.at(i);
        keyword_hit_flags.push_back(keyword_hit);
        bool inner_fist = true;
        for (;; ++j_begin) {
            bool flag = hit_flags.at(j_begin);
            if (inner_fist) {
                inner_fist = false;
                flag = flag && keyword_hit;
            }
            unified_token_hit_flags.push_back(flag);
            if (answers.at(i) == answers.at(j_begin) + 1 ||
                    (answers.at(i) == 0 && answers.at(i) == answers.at(j_begin))) {
                cout << "k ";
                if (keyword_hit && hit_flags.at(j_begin)) {
                    ++hit_count;
                }
                ++j_begin;
                break;
            } else {
                if (hit_flags.at(j_begin)) {
                    ++hit_count;
                }
                cout << "y ";
            }
        }
    }
    cout << endl;

    if (len != unified_token_hit_flags.size()) {
        cerr << boost::format("sentence len:%1% unified_token_hit_flags size:%2%") % len %
            unified_token_hit_flags.size() << endl;
        abort();
    }

    hit_count_result = hit_count;
    return log_sum;
}

#endif
