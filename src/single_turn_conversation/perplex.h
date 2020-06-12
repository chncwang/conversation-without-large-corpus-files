#ifndef SINGLE_TURN_CONVERSATION_PERPLEX_H
#define SINGLE_TURN_CONVERSATION_PERPLEX_H

#include <memory>
#include <tuple>
#include <utility>
#include <vector>
#include <cmath>
#include <iostream>
#include <boost/format.hpp>

#include "N3LDG.h"

float computePerplex(const std::vector<Node *> &nodes, const std::vector<int> &answers, int len,
        int &hit_count_result,
        vector<int> &keyword_hit_flags,
        vector<int> &token_hit_flags,
        vector<int> &unified_token_hit_flags) {
    float log_sum = 0.0f;
    keyword_hit_flags.clear();
    token_hit_flags.clear();
    unified_token_hit_flags.clear();
    vector<int> hit_flags;

    for (int i = 0; i < nodes.size(); ++i) {
        Node &node = *nodes.at(i);
        dtype reciprocal_answer_prob = 1 / node.getVal().v[answers.at(i)];
        log_sum += log(reciprocal_answer_prob);
        bool hit = true;
        for (int j = 0; j < node.getDim(); ++j) {
            if (node.getVal().v[j] > node.getVal().v[answers.at(i)]) {
                hit = false;
                break;
            }
        }
        hit_flags.push_back(hit);
        if (i < len) {
            token_hit_flags.push_back(hit);
        }
    }

    int hit_count = 0;
    int j_begin = 0;

    for (int i = len; i < nodes.size(); ++i) {
        bool keyword_hit = hit_flags.at(i);
        keyword_hit_flags.push_back(keyword_hit);
        unified_token_hit_flags.push_back(keyword_hit);
        for (;; ++j_begin) {
            unified_token_hit_flags.push_back(hit_flags.at(j_begin));
            if (answers.at(i) == answers.at(j_begin) + 1 ||
                    (answers.at(i) == 0 && answers.at(i) == answers.at(j_begin))) {
                cout << "k ";
                if (keyword_hit && hit_flags.at(j_begin)) {
                    ++hit_count;
                }
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

    hit_count_result = hit_count;
    return log_sum;
}

#endif
