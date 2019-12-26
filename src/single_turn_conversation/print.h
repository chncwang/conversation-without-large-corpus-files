#ifndef CONVERSATION_PRINT_H
#define CONVERSATION_PRINT_H

#include <string>
#include <vector>
#include <iostream>

using namespace std;

void print(const vector<string> &words) {
    for (const string &w : words) {
        cout << w << " ";
    }
    cout << endl;
}


#endif
