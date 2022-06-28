#pragma once
#include "pch.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue> 
#include <stack> 
#include <string>
#include <cstring>
#include <algorithm>    // std::reverse
#include <unordered_set>
#include <bitset>
using namespace std;




class Solution {
public:

	//190106
	struct ListNode {
		int val;
		ListNode* next;
		ListNode() : val(0), next(nullptr) {}
		ListNode(int x) : val(x), next(nullptr) {}
		ListNode(int x, ListNode* next) : val(x), next(next) {}
	};

	struct TreeNode {
		int val;
		TreeNode* left;
		TreeNode* right;
		TreeNode() : val(0), left(nullptr), right(nullptr) {}
		TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
		TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}

	};

	//220623 build a binary search tree (BST) from level-first order.
	template <typename T>
	T* build_b_tree_level_order(vector<T>& node_vector, T* root);

	//181230
	vector<int> twoSum(vector<int>& nums, int target);
	//181231
	int reverse(int x);
	//190102
	bool isPalindrome(int x);
	//190103
	int romanToInt(string s);
	//190104
	string longestCommonPrefix(vector<string>& strs);
	//190105
	bool isValid(string s);
	//190106
	struct ListNode;
	ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
	//190107
	int removeDuplicates(vector<int>& nums);
	//190108
	int removeElement(vector<int>& nums, int val);
	//190114
	bool checkPossibility(vector<int>& nums);
	//190115
	int largestPalindrome(int n);
	//190116
	bool buddyStrings(string A, string B);
	//190117
	int countPrimes(int n);
	//190118
	void rotate(vector<int>& nums, int k);
	//190121
	string convertToTitle(int n);
	//190122
	bool isBadVersion(int n);
	int firstBadVersionRec(int L, int R);
	//190221
	int findPairs(vector<int>& nums, int k);
	//190222
	int findUnsortedSubarray(vector<int>& nums);
	//190223
	int findNthDigit(int n);
	//190224
	uint32_t reverseBits(uint32_t n);
	//190227
	bool isPalindrome(string s);
	//190303
	int mySqrt(int x);
	//190304
	bool canPlaceFlowers(vector<int>& flowerbed, int n);
	//190310
	int findRadius(vector<int>& houses, vector<int>& heaters);
	//190316
	int repeatedStringMatch(string A, string B);
	//220222
	bool isPowerOfThree(int n);
	//220225
	int strStr(string haystack, string needle);
	//220412
	string countAndSay(int n);
	string count_and_say(int i, int& n, string temp);
	//220413
	void setZeroes(vector<vector<int>>& matrix);
	//220415
	struct Node138;
	Node138* copyRandomList(Node138* head);
	//
	//void GetResult(int* p, int& Get_Result);

	//void get_result(vector<vector<int>>& v, int& sum_m, int sum, int& Get_Result);
	int huawei0();
	//220420
	int trap(vector<int>& height);
	void recursive(vector<int>& height, stack<int> s, int r, int& result);

	//220421
	int rob(vector<int>& nums);
	int job_recursive(vector<int>& nums, int k, vector<int>& dp);

	void dfs_order(int& i, map<int, vector<int>>& G, vector<bool>& V, vector<bool>& P, stack<int>& S, bool& cycle) {

		if (cycle) return;

		V[i] = true;
		P[i] = true;

		for (int j : G[i]) {
			if (P[j])
				cycle = true;
			if (!V[j])
				dfs_order(j, G, V, P, S, cycle);
		}

		S.emplace(i);
		P[i] = false;
		/*auto it = mp.begin();
		while (it != mp.end()) {
			for (int i = 0; i < it->second.size(); i++) {
				if (!mp[it->second[i]].empty()) {
					dfs_order(mp, mp[it->second[i]]);
				}
				else {

				}
			}
		}*/
	}

	vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
		int n = prerequisites.size();

		if (numCourses == 1 && n == 0) {
			return { 0 };
		}

		if (numCourses == 2 && n == 0) {
			return { 1,0 };
		}

		vector<int> Ans;
		vector<bool> V(numCourses, false);
		vector<int> degree(numCourses, 0);

		map<int, vector<int>> G;
		queue<int> q;

		for (int i = 0; i < n; i++) {
			G[prerequisites[i][1]].emplace_back(prerequisites[i][0]);
		}

		for (auto adj : G) {
			for (int v : adj.second) {
				degree[v]++;
			}
		}

		for (int i = 0; i < degree.size(); i++) {
			if (degree[i] == 0) {
				q.emplace(i);
			}
		}

		while (!q.empty()) {
			int cur = q.front();
			q.pop();
			Ans.emplace_back(cur);

			for (auto au : G[cur]) {
				degree[au]--;
				if (degree[au] == 0) {
					q.emplace(au);
				}
			}
		}

		if (Ans.size() == numCourses) return Ans;

		return {};
	}


	int binary_search(vector<int>& nums, int& l, int& r) {
		int m = (l + r) / 2;

		if (l == m) return l;

		if (nums[m] < nums[m + 1]) {
			l = m + 1;
		}
		else {
			r = m;
		}
		return binary_search(nums, l, r);
	}

	int findPeakElement(vector<int>& nums) {
		int n = nums.size();
		int l = 0;
		int r = n - 1;
		int m = 0;
		/*while (l <= r) {
			m = (l + r) / 2;
			if (l == m) break;
			if (nums[m] < nums[m+1]) {
				l = m + 1;
			}
			else {
				r = m;
			}
		}*/

		m = binary_search(nums, l, r);

		if (m + 1 < n) {
			if (nums[m] > nums[m + 1]) {
				return m;
			}
			else {
				return m + 1;
			}
		}
		else {
			return l;
		}
	}

	bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
		if (numCourses == 1) return true;

		vector<vector<int>> G(numCourses);
		vector<bool> done(numCourses, false);
		vector<bool> todo(numCourses, false);
		vector<int> T(numCourses, 0);
		//int count = 0;
		bool cycle = true;

		for (auto&& au : prerequisites) {
			G[au[1]].emplace_back(au[0]);
		}

		for (int i = 0; i < numCourses; i++) {
			//if(!done[i] && !dfs_topo(i, G, done, todo, T, cycle))
				//return false;
			if (is_cycle(i, G, done, todo, T, cycle))
				return false;

		}

		return true;
	}

	bool is_cycle(int i, vector<vector<int>>& G, vector<bool>& done, vector<bool>& todo, vector<int>& T, bool& cycle) {
		/*if (done[i]) return true;

		if (todo[i]) return false;

		done[i] = todo[i] = true;*/
		if (T[i] == 1) return true;

		if (T[i] == 0) {
			T[i] = 1;
			for (int j = 0; j < G[i].size(); j++) {
				if (is_cycle(G[i][j], G, done, todo, T, cycle))
					return true;
			}
		}

		T[i] = 2;
		//todo[i] = false;
		return false;
	}

	vector<vector<int>> merge(vector<vector<int>>& intervals) {
		int n = intervals.size();
		if (n == 1) return intervals;
		vector<vector<int>> result;
		vector<vector<int>> t;

		map<int, vector<int>> M;

		sort(intervals.begin(), intervals.end());

		for (auto au : intervals) {
			M[au.front()].emplace_back(au.back());
		}

		vector<int> temp(2, 0);

		for (auto it = M.begin(); it != M.end(); it++) {
			temp.front() = it->first;
			sort(it->second.begin(), it->second.end());
			temp.back() = it->second.back();
			t.emplace_back(temp);
		}

		int m = t.size();
		result.emplace_back(t.front());

		for (int i = 1; i < m; i++) {
			//temp.front() = result.back().front();
			if (result.back().back() >= t[i].front()) {
				result.back().back() = max(result.back().back(), t[i].back());
				continue;
			}
			else {
				//temp.back() = t[i].back();
				result.emplace_back(t[i]);
				continue;
			}
		}

		/*for (int i = 0; i < m; i++) {
			if (result.empty()) {
				temp.front() = t[i].front();
				if (i + 1 < m && t[i].back() >= t[i + 1].front()) {
					temp.back() = t[i + 1].back();
					if (t[i].back() >= t[i + 1].back()) {
						temp.back() = t[i].back();
					}
					i++;
					result.emplace_back(temp);
					continue;
				}
				else {
					temp.back() = t[i].back();
					result.emplace_back(temp);
					continue;
				}
			}
			else {
				temp.front() = result.back().front();
				if (i < m && result.back().back() >= t[i].front()) {
					temp.back() = t[i + 1].back();
					if (t[i].back() >= t[i + 1].back()) {
						temp.back() = t[i].back();
					}
					i++;
					result.emplace_back(temp);
					continue;
				}
				else {
					temp.back() = t[i].back();
					result.emplace_back(temp);
					continue;
				}
			}
		}*/


		return result;
	}

	int longestSubstring(string s, int k, int& result) {
		if (k == 0) return 0;
		//if (s.size() < k) return 0;

		map<char, int> M;

		for (int i = 0; i < s.size(); i++) {
			M[s[i]]++;
		}

		int count = 0;

		for (auto au : M) {
			if (au.second >= k) {
				count++;
			}
		}

		int n = s.size();

		int id = 0;
		while (id < n && M[s[id]] >= k) {
			id++;
		}

		if (id == n) return n;

		//int result = 0;

		//for (int i = 1; i < n; i++) {
			//string s1 = s.substr(0, i);
			//string s2 = s.substr(i);

			//if (s1=="aaa") {
				//cout << s2;
			//}

		result = max(longestSubstring(s.substr(0, id), k, result), longestSubstring(s.substr(id + 1), k, result));
		//}

		return result;
		//return max(longestSubstring(s.substr(0, id), k), longestSubstring(s.substr(id+1, n), k));

	}

	bool wordBreak(string s, vector<string>& wordDict) {
		int n = s.size();
		vector<bool> dp(n + 1);
		dp[0] = true;

		for (int i = 0; i < n; i++) {
			for (int j = n - i; j > 0; j--) {

				if (dp[i]) {
					string ts = s.substr(i, j);
					if (find(wordDict.begin(), wordDict.end(), ts) != wordDict.end()) {
						dp[i + j] = true;
					}
				}
			}
		}

		return dp[s.size()];
	}

	void ffind_result(string s, vector<string>& wordDict, bool& r) {
		int n = s.size();
		int m = wordDict.size();

		if (n == 0) {
			r = true;
			return;
		}

		for (int i = 0; i < m; i++) {
			string ts = s.substr(0, wordDict[i].size());
			if (ts == wordDict[i]) {
				ffind_result(s.substr(ts.size()), wordDict, r);
			}
		}
	}

	void find_result(vector<vector<string>>& ss, int i, bool& r) {
		int n = ss[0].size();
		for (int j = 0; j < n; j++) {
			if (ss[i][j] != "") {
				if (j == n - 1) {
					r = true;
					return;
				}
				find_result(ss, j + 1, r);
			}
		}
	}

	vector<int> partitionLabels(string s) {
		size_t n = s.size();
		vector<int> r;
		vector<char> t{ s[0] };
		vector<vector<char>> tt{ t };
		map<char, int> dic;
		dic[s[0]]++;

		for (int i = 1; i < n; i++) {
			vector<char> ttt;
			if (dic[s[i]]) {

				for (int j = 0; j < tt.size(); j++) {
					if (find(tt[j].begin(), tt[j].end(), s[i]) != tt[j].end()) {
						tt.back().emplace_back(s[i]);
						int m = tt.size() - 1;
						while (m > j) {
							tt[j].insert(tt[j].end(), tt[m].begin(), tt[m].end());
							tt.erase(tt.begin() + m);
							m--;
						}

						break;
					}
				}
			}
			else {
				ttt.emplace_back(s[i]);
				tt.emplace_back(ttt);
				dic[s[i]]++;
			}

		}

		for (int j = 0; j < tt.size(); j++) {
			r.emplace_back(tt[j].size());
		}

		return r;

	}

	//220518 739. Daily Temperatures
	vector<int> dailyTemperatures(vector<int>& temperatures) {
		size_t n = temperatures.size();
		vector<int> r(n, 0);
		stack<int> s;

		int count = 0;
		for (int i = 0; i < n; i++) {
			while (!s.empty() && temperatures[s.top()] < temperatures[i]) {
				r[s.top()] = i - s.top();
				s.pop();
			}
			s.emplace(i);
		}

		/*for (int i = 0; i < n-1; i++) {
			int count = 0;
			for (int j = i + 1; j < n; j++) {
				count++;
				if (temperatures[i] < temperatures[j]) {
					r[i]+=count;
					break;
				}
			}
		}*/
		return r;
	}

	/*===== 220519 39. Combination Sum ======*/
	vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
		vector<int> tr;
		vector<vector<int>> r;
		int begin = 0;

		for (int i = 0; i < candidates.size(); i++) {
			tr.emplace_back(candidates[i]);
			back_track(candidates, target, i, tr, r);
			tr.pop_back();
		}

		return r;
	}

	void back_track(vector<int>& candidates, int& target, int begin, vector<int>& tr, vector<vector<int>>& r) {
		int temp = 0;

		for (int i = 0; i < tr.size(); i++) {
			temp += tr[i];
		}

		if (temp == target) {
			r.emplace_back(tr);
			return;
		}
		else if (temp > target) {
			return;
		}

		for (int i = begin; i < candidates.size(); i++) {
			tr.emplace_back(candidates[i]);
			back_track(candidates, target, i, tr, r);
			tr.pop_back();
		}
	}
	/*================================================*/
	/*===== 220520 64. Minimum Path Sum ======*/
	int minCost(vector<vector<int>>& cost, int m, int n) {
		cout << m << ", " << n << endl;
		if (n < 0 || m < 0)
			return INT_MAX;
		else if (m == 0 && n == 0)
			return cost[m][n];

		return cost[m][n] + min(minCost(cost, m - 1, n), minCost(cost, m, n - 1));
	}

	int minPathSum(vector<vector<int>>& grid) {
		int M = grid.size(), N = grid[0].size();
		return minCost(grid, M - 1, N - 1);
	}

	void back_track_64(vector<vector<int>>& grid, int begin_row, int begin_colume, int temp_sum, vector<int>& sum) {

		size_t m = grid.size();
		size_t n = grid[0].size();

		if (begin_row == m - 1 && begin_colume == n - 1) {
			sum.emplace_back(temp_sum);
		}

		if (begin_row > m - 1 || begin_colume > n - 1) {
			return;
		}
		if (begin_row + 1 < m) {
			temp_sum += grid[begin_row + 1][begin_colume];
			back_track_64(grid, begin_row + 1, begin_colume, temp_sum, sum);
			temp_sum -= grid[begin_row + 1][begin_colume];
		}
		if (begin_colume + 1 < n) {
			temp_sum += grid[begin_row][begin_colume + 1];
			back_track_64(grid, begin_row, begin_colume + 1, temp_sum, sum);
		}
	}
	/*================================================*/
	/*===== 220521 24. Swap Nodes in Pairs ======*/
	ListNode* swapPairs(ListNode* head) {

		if (!head) {
			return nullptr;
		}
		else if (!head->next) {
			return head;
		}

		ListNode* h = head->next;
		ListNode* t = head;
		ListNode* n = head->next;
		ListNode* p = head;
		ListNode* pr = nullptr;

		while (p && p->next) {

			if (!p->next) break;

			t = p;
			n = p->next;
			if (pr)
				pr->next = n;

			t->next = n->next;
			n->next = t;

			p = t->next;
			pr = t;


		}


		return h;
	}
	/*================================================*/
	/*====  96. Unique Binary Search Trees  ====*/
	int numTrees(int n) {
		vector<int> t(n + 1, 0);
		t[0] = t[1] = 1;
		int r = 0;
		//r = dp_tree(n,t);

		for (int i = 2; i <= n; i++) {
			for (int j = 0; j < i; j++) {
				t[i] += t[j] * t[i - j - 1];
			}
		}


		return t[n];
	}

	int dp_tree(int n, vector<int>& t) {

		if (n == 1 || n == 0) {
			return 1;
		}
		if (t[n] != -1) {
			return t[n];
		}
		int rr = 0;
		for (int i = 0; i < n; i++) {
			rr += dp_tree(i, t) * dp_tree(n - i - 1, t);
		}

		return t[n] = rr;
	}
	/*================================================*/

	/*==== 114. Flatten Binary Tree to Linked List  ====*/
	void flatten(TreeNode* root) {

		if (!root) return;
		if (!root->left && !root->right) return;

		vector<TreeNode*> result = { root };

		dfs_114(root, result);
		TreeNode* t = result[0];
		for (int i = 1; i < result.size(); i++) {
			t->left = nullptr;
			t->right = result[i];
			t = t->right;
		}

		root = result[0];

		return;
	}


	void dfs_114(TreeNode* node, vector<TreeNode*>& result) {

		if (node->left) {
			result.emplace_back(node->left);
			dfs_114(node->left, result);
		}

		if (node->right) {
			result.emplace_back(node->right);
			dfs_114(node->right, result);
		}

		return;
	}
	/*================================================*/
	/*==== 394. Decode String ====*/

	string decodeString(const string& s, int& i) {
		string res;
		while (i < s.size() && s[i] != ']') {
			if (!isdigit(s[i])) {
				res += s[i];
				i++;
			}
			else {
				string nn;
				while (isdigit(s[i])) {
					nn += s[i];
					i++;
				}
				int n = stoi(nn);
				i++;
				string t = decodeString(s, i);
				i++;

				while (n-- > 0) {
					res += t;
				}

			}

		}

		return res;
	}

	string decodeString(string s) {
		int i = 0;
		return decodeString(s, i);
	}
	/*================================================*/
	/*===== 134. Gas Station =====*/

	int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
		size_t n = gas.size();
		int tank = 0;
		int begin = -1;
		int count = 0;
		int cs = 0;
		int ns = 0;
		for (int i = 0; i < n; i++) {
			cs += gas[i] - cost[i];
			ns += gas[i] - cost[i];
			if (ns < 0) {
				begin = i + 1;
				ns = 0;
			}

			/*int od = i % n;
			tank += gas[od] - cost[od];
			count++;
			if (tank < 0) {
				tank = 0;
				count = 0;
			}
			if (count == n) {
				begin = (i + 1) % n;
			}*/

			//tank = begin_circle(gas, cost, i);
			//if (tank >= 0) {
				//return i;
			//}
		}
		if (cs < 0) {
			begin = -1;
		}

		return begin;
	}

	int begin_circle(vector<int>& gas, vector<int>& cost, int begin) {
		size_t n = gas.size();

		int tank = gas[begin];

		for (int j = begin; j < begin + n; j++) {
			int od = (j) % n;
			int nex = (j + 1) % n;

			if (tank < cost[od]) {
				return -1;
			}

			tank += gas[nex] - cost[od];
			if (tank < 0) {
				return tank;
			}
		}

		return tank;
	}
	/*================================================*/

	/*===== 150. Evaluate Reverse Polish Notation =====*/
	int evalRPN(vector<string>& tokens) {
		int n = tokens.size();

		if (n == 1) {
			return stoi(tokens.front());
		}

		int result = 0;
		map<char, int> operator_map{ {'+', 0}, { '-', 1 }, { '*', 2 }, { '/', 3 } };
		map<string, int> operator_mmap{ {" + ", 0}, { " - ", 1 }, { " * ", 2 }, { " / ", 3 } };
		stack<int> digit;
		stack<char> opera;
		digit.emplace(stoi(tokens[0]));
		digit.emplace(stoi(tokens[1]));

		//recursive_150(tokens, operator_map, 0);

		/*---------------------------   Recursive   -------------------------------------------*/
		int m = n - 1;
		result = resursive_150(tokens, operator_map, m);
		return result;
		/*-------------------------------------------------------------------------------------*/

		/*---------------------------   Stack   ------------------------------------------------*/
		/*

		int i = 2;
		while (!digit.empty() && i < n) {
			if (is_operator(tokens[i])) {
				int right = digit.top();
				digit.pop();
				int left = digit.top();
				digit.pop();

				digit.emplace( operation(tokens, operator_map, i, left, right) );
				i++;
			}
			else {
				digit.emplace(stoi(tokens[i]));
				i++;
			}
		}*/
		/*-------------------------------------------------------------------------------------*/

		return digit.top();
	}

	int resursive_150(vector<string>& tokens, map<char, int>& operator_map, int& i) {
		/*if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/") {
			char Operator = tokens[i][0];
			int op1 = resursive_150(tokens, operator_map, --i);
			int op2 = resursive_150(tokens, operator_map, --i);
			if (Operator == '+') return op2 + op1;
			if (Operator == '-') return op2 - op1;
			if (Operator == '/') return op2 / op1;
			if (Operator == '*') return op2 * op1;
		}
		return stoi(tokens[i]);*/

		if (is_operator(tokens[i])) {
			int j = i;
			int right = resursive_150(tokens, operator_map, --i);
			int left = resursive_150(tokens, operator_map, --i);
			return operation(tokens, operator_map, j, left, right);
		}

		return stoi(tokens[i]);
	}

	int operation(vector<string>& tokens, map<char, int>& operator_map, int i, int left, int right) {
		int result = 0;

		switch (operator_map[tokens[i][0]]) {
		case 0:
			return result = left + right;
			break;
		case 1:
			return result = left - right;
			break;
		case 2:
			return result = left * right;
			break;
		case 3:
			return result = left / right;
			break;
		default:
			assert(0);
		}
	}

	bool is_operator(string& token) {
		if (token.size() == 1 && !isdigit(token[0])) {
			return true;
		}

		return false;
	}

	/*================================================*/


	/*====227. Basic Calculator II=======*/
	int calculate(string s) {
		stack<int> myStack;
		char sign = '+';
		int res = 0, tmp = 0;
		for (unsigned int i = 0; i < s.size(); i++) {
			if (isdigit(s[i]))
				tmp = 10 * tmp + s[i] - '0';
			if (!isdigit(s[i]) && !isspace(s[i]) || i == s.size() - 1) {
				if (sign == '-')
					myStack.push(-tmp);
				else if (sign == '+')
					myStack.push(tmp);
				else {
					int num;
					if (sign == '*')
						num = myStack.top() * tmp;
					else
						num = myStack.top() / tmp;
					myStack.pop();
					myStack.push(num);
				}
				sign = s[i];
				tmp = 0;
			}
		}
		while (!myStack.empty()) {
			res += myStack.top();
			myStack.pop();
		}
		return res;
	}

	int get_int(string s, int& i) {
		size_t n = s.size();
		string t_s;
		while (i < n && isdigit(s[i])) {
			t_s += s[i];
			i++;
		}
		int right = stoi(t_s);
		return right;
	}

	int recursive_227(string& s, int& i) {
		size_t n = s.size();
		int reuslt = 0;
		update_i(s, i);
		string t_s;
		while (i >= 0 && isdigit(s[i])) {
			t_s += s[i];
			i--;
		}
		std::reverse(t_s.begin(), t_s.end());
		int right = stoi(t_s);
		update_i(s, i);
		if (i <= 0) {
			return reuslt = right;
		}

		if (s[i] == '*') {
			i--;
			update_i(s, i);
			string t_s;
			while (i >= 0 && isdigit(s[i])) {
				t_s += s[i];
				i--;
			}
			std::reverse(t_s.begin(), t_s.end());
			int left = stoi(t_s);
			update_i(s, i);
			if (i <= 0) {
				return reuslt = left * right;
			}
			update_i(s, i);
			if (s[i] == '*')      return reuslt = recursive_227(s, --i) * left * right;
			else if (s[i] == '/') return reuslt = floor(recursive_227(s, --i) / left) * right;
			else if (s[i] == '+') return reuslt = recursive_227(s, --i) + left * right;
			else if (s[i] == '-') return reuslt = recursive_227(s, --i) - left * right;
		}

		if (s[i] == '/') {
			i--;
			update_i(s, i);
			string t_s;
			while (i >= 0 && isdigit(s[i])) {
				t_s += s[i];
				i--;
			}
			std::reverse(t_s.begin(), t_s.end());
			int left = stoi(t_s);
			update_i(s, i);
			if (i <= 0) {
				return reuslt = floor(left / right);
			}
			update_i(s, i);
			if (s[i] == '*')      return reuslt = floor(recursive_227(s, --i) * left / right);
			else if (s[i] == '/') return reuslt = floor(floor(recursive_227(s, --i) / left) / right);
			else if (s[i] == '+') return reuslt = recursive_227(s, --i) + floor(left / right);
			else if (s[i] == '-') return reuslt = recursive_227(s, --i) - floor(left / right);
		}

		if (s[i] == '+') {
			i--;
			update_i(s, i);
			string t_s;
			while (i >= 0 && isdigit(s[i])) {
				t_s += s[i];
				i--;
			}
			std::reverse(t_s.begin(), t_s.end());
			int left = stoi(t_s);
			update_i(s, i);
			if (i <= 0) {
				//return reuslt = floor(left + right);

				if (s[i] == '*') return reuslt = recursive_227(s, --i) * left + right;
				else if (s[i] == '/') { return reuslt = floor(recursive_227(s, --i) / left) + right; }
				else if (s[i] == '+') { return reuslt = recursive_227(s, --i) + left + right; }
				else if (s[i] == '-') { return reuslt = recursive_227(s, --i) - left + right; }
			}
			update_i(s, i);

			if (s[i] == '*') return reuslt = recursive_227(s, --i) * left + right;
			else if (s[i] == '/') { return reuslt = floor(recursive_227(s, --i) / left) + right; }
			else if (s[i] == '+') { return reuslt = recursive_227(s, --i) + left + right; }
			else if (s[i] == '-') { return reuslt = recursive_227(s, --i) - left + right; }
		}

		if (s[i] == '-') {
			i--;
			update_i(s, i);
			string t_s;
			while (i >= 0 && isdigit(s[i])) {
				t_s += s[i];
				i--;
			}
			std::reverse(t_s.begin(), t_s.end());
			int left = stoi(t_s);
			update_i(s, i);
			if (i <= 0) {
				return reuslt = floor(left - right);
			}
			update_i(s, i);

			if (s[i] == '*')      return reuslt = recursive_227(s, --i) * left - right;
			else if (s[i] == '/') return reuslt = floor(recursive_227(s, --i) / left) - right;
			else if (s[i] == '+') return reuslt = recursive_227(s, --i) + left - right;
			else if (s[i] == '-') return reuslt = recursive_227(s, --i) - left - right;
		}

		return reuslt;
	}

	void update_i(string& s, int& i) {
		size_t n = s.size();
		while (i < n && s[i] == ' ') {
			i++;
		}
	}
	/*================================================*/


	/*===============   54. Spiral Matrix  ============*/
	vector<int> spiralOrder(vector<vector<int>>& matrix) {
		int b_n = 0;
		int b_m = 0;
		size_t n = matrix.size();
		size_t m = matrix[0].size();
		vector<int> r;
		if (n == 1 || m == 1) {
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < m; j++) {
					r.emplace_back(matrix[i][j]);
				}
			}
			return r;
		}
		while (b_m < m && b_n < n) {
			for (int j = b_m; j < m; j++) {
				r.emplace_back(matrix[b_n][j]);
			}
			for (int i = b_n + 1; i < n; i++) {
				r.emplace_back(matrix[i][m - 1]);
			}
			for (int j = m - 2; j >= b_m && b_n + 1 < n; j--) {
				r.emplace_back(matrix[n - 1][j]);
			}
			for (int i = n - 2; i > b_n && b_m + 1 < m; i--) {
				r.emplace_back(matrix[i][b_m]);
			}
			b_m++;
			b_n++;
			m--;
			n--;
		}

		return r;
	}

	/*================================================*/


	/*====== 334. Increasing Triplet Subsequence ======*/
	bool increasingTriplet(vector<int>& nums) {
		size_t n = nums.size();
		if (n < 3) {
			return false;
		}

		int left = nums[0];
		int middle = nums[0];
		int right = nums[0];

		for (int i = 1; i < n; i++) {
			if (nums[i] > left) {
				if (nums[i] > middle && left != middle) {
					return true;
				}
				else {
					left = right;
					middle = nums[i];
				}
				continue;
			}
			else if (nums[i] > right && right < left) {
				left = right;
				middle = nums[i];
				continue;
			}
			else if (nums[i] < left && left >= right) {
				right = nums[i];
				continue;
			}
		}

		return false;
	}
	/*================================================*/

	/*========  172. Factorial Trailing Zeroes ========*/
	int trailingZeroes(int n) {
		if (n < 5) return 0;
		int r = 0;
		long int number = 1;

		for (int i = 2; i <= n; i++) {
			number *= i;
		}

		while (number % 10 == 0) {
			r++;
			number /= 10;
		}

		return r;
	}
	/*================================================*/


	/*============== 322. Coin Change ================*/

	int coinChange(vector<int>& coins, int amount) {
		int r = 0;
		size_t n = coins.size();

		if (!amount) {
			return 0;
		}

		vector<int> dp(amount + 1, amount + 1);
		dp[0] = 0;

		for (int i = 1; i <= amount; i++) {
			for (int j = 0; j < n; j++) {
				if (coins[j] <= i) {
					dp[i] = min(dp[i], dp[i - coins[j]] + 1);
				}
			}
		}

		r = dp[amount];

		if (r == amount + 1) r = -1;

		return r;
	}

	int dp_322(vector<int>& coins, int i, int j, int amount, int r, vector<vector<int>>& r_vector) {
		size_t n = coins.size();

		if (amount < 0) return INT_MAX * -1;
		if (amount == 0) {
			r_vector[i][j] = r;
			return r;
		}

		for (int i = 0; i < n; i++) {
			r_vector[i][0] += dp_322(coins, i, 0, amount, r, r_vector);
			r_vector[i][1] += dp_322(coins, i, 1, amount - coins[i], r + 1, r_vector);
		}
	}
	/*================================================*/


	/*=======  34. Find First and Last Position of Element in Sorted Array  ===========*/
	//220531
	vector<int> searchRange(vector<int>& nums, int target) {

		size_t n = nums.size();
		int b = 0;
		int e = n - 1;
		vector<int> r = { -1, -1 };

		if (nums.empty()) {
			return vector<int>{-1, -1};
		}

		while (b < e) {
			int m = (b + e) / 2;

			if (nums[m] < target) {
				b = m + 1;
			}
			else if (nums[m] > target) {
				e = m - 1;
			}
			else {//==target
				b = e = m;
				while (e + 1 < n && nums[e + 1] == target) {
					e++;
				}
				while (b - 1 > -1 && nums[b - 1] == target) {
					b--;
				}
				break;
			}

		}

		if ((b == e && nums[e] != target) || b > e) {

			return vector<int>{-1, -1};

		}

		r = { b, e };

		return r;
	}


	/*================================================*/

	/*==============    79. Word Search    ==============*/
	bool exist(vector<vector<char>>& board, string word) {
		int n = board.size();
		int m = board[0].size();
		int i = 0;
		int j = 0;
		int k = 0;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {

				if (back_track_79(board, word, i, j, k)) {
					return true;
				}
			}
		}

		return false;
	}

	bool back_track_79(vector<vector<char>>& board, string word, int i, int j, int k) {

		int n = board.size();
		int m = board[0].size();
		int k_size = word.size();

		if (k == k_size) {
			return true;
		}

		if (i < 0 || i >= n || j < 0 || j >= m || board[i][j] != word[k]) {
			return false;
		}

		bool res = false;
		char t = board[i][j];
		board[i][j] = '*';

		res = back_track_79(board, word, i - 1, j, k + 1) || back_track_79(board, word, i + 1, j, k + 1) ||
			back_track_79(board, word, i, j - 1, k + 1) || back_track_79(board, word, i, j + 1, k + 1);

		board[i][j] = t;

		return res;
	}
	/*===================================================*/

	/*==============    2. Add Two Numbers 220605  ==============*/
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		vector<int> num;
		ListNode* h1 = l1;
		ListNode* h2 = l2;

		int next_value = 0;

		recursion_79(h1, h2, next_value, num);

		ListNode* p3 = new ListNode(num[0]);
		ListNode* h3 = p3;
		for (auto au : num) {
			p3->next = new ListNode(au);
			p3 = p3->next;
		}
		h3 = h3->next;

		return h3;
	}

	void recursion_79(ListNode* n1, ListNode* n2, int next_value, vector<int>& num) {
		if (n1 && n2) {
			int sum_val = n1->val + n2->val + next_value;
			if (sum_val > 9) {
				next_value = 1;
				int temp_val = sum_val % 10;
				num.emplace_back(temp_val);
			}
			else {
				next_value = 0;
				num.emplace_back(sum_val);
			}
			recursion_79(n1->next, n2->next, next_value, num);
		}
		else if (n1) {
			int sum_val = n1->val + next_value;
			if (sum_val > 9) {
				next_value = 1;
				int temp_val = sum_val % 10;
				num.emplace_back(temp_val);
			}
			else {
				next_value = 0;
				num.emplace_back(sum_val);
			}
			recursion_79(n1->next, n2, next_value, num);
		}
		else if (n2) {
			int sum_val = n2->val + next_value;
			if (sum_val > 9) {
				next_value = 1;
				int temp_val = sum_val % 10;
				num.emplace_back(temp_val);
			}
			else {
				next_value = 0;
				num.emplace_back(sum_val);
			}
			recursion_79(n1, n2->next, next_value, num);
		}
		else if (next_value) {
			num.emplace_back(next_value);
			return;
		}
	}
	/*===================================================*/

	/*======  19. Remove Nth Node From End of List  ======*/
	ListNode* removeNthFromEnd(ListNode* head, int n) {
		ListNode* p1 = head;
		ListNode* p2 = head;
		int m = 0;

		while (p1) {
			m++;
			p1 = p1->next;
		}

		p1 = head;

		while (m > n) {
			m--;
			p1 = p2;
			p2 = p2->next;
		}

		if (p1 == p2) {
			return head = head->next;
		}

		if (p2->next) {
			p1->next = p2->next;
		}
		else {
			p1->next = nullptr;
		}

		return head;
	}
	/*====================================================*/

	/*======  33. Search in Rotated Sorted Array  ======*/
	int search(vector<int>& nums, int target) {

		int n = nums.size();
		int l = 0;
		int r = n - 1;

		if (l == r && nums[0] != target) {
			return -1;
		}
		else if (l == r && nums[0] == target) {
			return 0;
		}

		if (nums[l] > nums[r] && nums[l] > target && nums[r] < target) {
			return -1;
		}

		while (l <= r) {
			int m = (l + r) / 2;

			if (nums[l] < nums[r] && nums[m] < target) {
				l = m + 1;
			}
			else if (nums[l] < nums[r] && nums[m] > target) {
				r = m - 1;
			}
			else if (nums[l] > nums[r] && nums[m] < target) {
				if (nums[r] < target && nums[m] < nums[r]) {
						r = m - 1;
				}
				else {
					l = m + 1;
				}

			}
			else if (nums[l] > nums[r] && nums[m] > target) {
				if (nums[l] > target && nums[m] >= nums[l]) {
						l = m + 1;
				}
				else {
					r = m - 1;
				}
			}
			else if (nums[m] == target) {
				return m;
			}
			else {
				return -1;
			}
		}

		return -1;
	}
	/*====================================================*/

	/*=================  55. Jump Game   =================*/
	bool canJump(vector<int>& nums) {
		int n = nums.size();
		bool r = false;

		if (n == 1 || nums[0] >= n - 1) {
			return true;
		}

		vector<int> dp(n, 0);

		int last_index = 0;
		for (int i = 0; i < n; i++) {
			if (last_index < i) {
				return false;
			}

			if (i + nums[i] >= n - 1) {
				return true;
			}

			dp[i + nums[i]] = max(dp[i + nums[i]], i + nums[i]);

			last_index = max(dp[i + nums[i]], last_index);
		}

		return dp[n - 1];
	}

	/*====================================================*/

	/*============ 152. Maximum Product Subarray =========*/
	int maxProduct(vector<int>& nums) {
		int n = nums.size();
		if (n == 1) return nums[0];

		vector<int> r(n, -INT_MAX);
		r[0] = nums[0];
		int temp_product = nums[0];
		int min_nagative = nums[0] < 0 ? nums[0] : 0;

		int min_pro = 1;
		int max_pro = 1;
		int res = nums[0];

		for (int i = 0; i < n; i++) {
			if (nums[i] < 0) {
				swap(min_pro, max_pro);
			}

			min_pro = min(nums[i], min_pro * nums[i]);
			max_pro = max(nums[i], max_pro * nums[i]);
			res = max(res, max_pro);

		}

		cout << res << endl;

		for (int i = 1; i < n; i++) {

			if (temp_product * nums[i] > 0) {
				if (nums[i] > 0) {
					temp_product *= nums[i];
					min_nagative = min(min_nagative * nums[i], 0);
				}
				else {
					temp_product = min_nagative * nums[i];
					min_nagative = nums[i];
				}
			}
			else if (temp_product * nums[i] < 0) {

				if (temp_product > 0 && nums[i] < 0) {

					if (min_nagative == 0) {
						min_nagative = temp_product * nums[i];
						temp_product = nums[i];
					}
					else {
						int temp = temp_product;
						temp_product = min_nagative * nums[i];
						min_nagative = temp * nums[i];
					}
				}
				else if(temp_product < 0 && nums[i] > 0) {
					temp_product = nums[i];
					min_nagative *= nums[i];
				}
			}
			else {
				temp_product = nums[i];
				if (nums[i] > 0) {
					min_nagative = 0;
				}
				else {
					min_nagative = nums[i];
				}
			}

			r[i] = max(r[i - 1], temp_product);
		}

		int max_product_0 = r[n - 1];

		return max_product_0;
	}
	/*====================================================*/

	/*============     130. Surrounded Regions   =========*/

	void solve(vector<vector<char>>& board) {
		int m = board.size();
		int n = board[0].size();

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j]=='X') {
					continue;
				}
				else {
					dfs_is_border(board, i, j,-1,-1);
				}
			}
		}

		/*cout << "===================" << endl << endl;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				cout << board[i][j] << ", ";
			}
			cout << endl;
		}
		cout << endl;
		cout << "===================" << endl << endl;*/


		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'B') {
					if (i - 1 >= 0 && board[i - 1][j] == 'A') {
						board[i][j] = 'O';
						//return true;
					}
					else if (i + 1 < m && board[i + 1][j] == 'A') {
						board[i][j] = 'O';
						//return true;
					}
					else if (j - 1 >= 0 && board[i][j - 1] == 'A') {
						board[i][j] = 'O';
						//return true;
					}
					else if (j + 1 < n && board[i][j + 1] == 'A') {
						board[i][j] = 'O';
						//return true;
					}
					else {
						board[i][j] = 'X';
					}
				}
				else if(board[i][j] == 'A') {
					board[i][j] = 'O';
				}
				continue;
			}
		}

		//cout << board[0][0] << endl;
	}

	bool dfs_is_border(vector<vector<char>>& board, int i, int j, int ii, int jj) {
		//assert(board[i][j] == 'O');

		int m = board.size();
		int n = board[0].size();
		bool is_border = false;

		if (i < 0 || i > m - 1 || j < 0 || j > n - 1 || board[i][j] == 'X') {
			return false;
		}

		if (board[i][j] == 'B') {
			if (i - 1 >= 0 && board[i-1][j] == 'A') {
				board[i][j] = 'A';
				//return true;
			}
			else if (i + 1 < m && board[i + 1][j] == 'A') {
				board[i][j] = 'A';
				//return true;
			}
			else if (j - 1 >= 0 && board[i][j-1] == 'A') {
				board[i][j] = 'A';
				//return true;
			}
			else if (j + 1 < n && board[i][j + 1] == 'A') {
				board[i][j] = 'A';
				//return true;
			}
			else {
				return false;
			}
		}


		if (board[i][j] == 'O') {

			if(board[i][j] == 'O') board[i][j] = 'B';

			if (i == 0 || i == m - 1 || j == 0 || j == n - 1 || (ii> -1 && board[ii][jj] == 'A')) {
				board[i][j] = 'A';
			}

			bool a = false;
			bool b= false;
			bool c = false;
			bool d = false;

			if (i - 1 != ii) {
				a = dfs_is_border(board, i - 1, j,i,j);
			}

			if (i + 1 != ii) {
				b = dfs_is_border(board, i + 1, j, i, j);
			}

			if (j - 1 != jj) {
				c = dfs_is_border(board, i , j - 1, i, j);
			}

			if (j + 1 != jj) {
				d = dfs_is_border(board, i , j + 1, i, j);
			}

			is_border = a || b || c || d;

			if (is_border || board[i][j] == 'A') {
				board[i][j] = 'A';
				return true;
			}
			else {
				return false;
			}
		}

		return true;
	}

	/*====================================================*/


	/*=====   3. Longest Substring Without Repeating Characters   =========*/

	int lengthOfLongestSubstring(string s) {
		int r = 0;
		int n = s.size();

		if (s.empty()) {
			return r;
		}
		else if (s.size() == 1) {
			return 1;
		}

		map<char,int> repeat_char;
		vector<char> char_vector;
		int max_length = 1;

		char_vector.emplace_back(s[0]);
		repeat_char[s[0]]++;
		
		for (int i = 1; i < n; i++) {
			
			if(repeat_char[s[i]] > 0) {

				if (char_vector.back() == s[i]) {
					char_vector.clear();
					repeat_char.clear();
				}
				else{ 
					while (!char_vector.empty() && char_vector[0] != s[i]) {
						repeat_char[char_vector[0]] = 0;
						char_vector.erase(char_vector.begin());
					} 
				}

				if (!char_vector.empty() && char_vector[0] == s[i]) {
					char_vector.erase(char_vector.begin());
					repeat_char[s[i]] = 0;
				}

				char_vector.emplace_back(s[i]);
				repeat_char[s[i]]++;

				max_length = max(max_length, int(char_vector.size()));
			}
			else {
				char_vector.emplace_back(s[i]);
				max_length = max(max_length, int(char_vector.size()));
				repeat_char[s[i]]++;
			}
		}

		return max_length;
	}

	/*=====================================================================*/


	/*===============     179. Largest Number 220615    ===================*/
	string largestNumber(vector<int>& nums) {
		vector<string> s;
		string r;
		for (auto&& au: nums) {
			s.emplace_back(to_string(au));
		}

		sort(s.begin(), s.end(), [](string& s1, string& s2) { return s1 + s2 > s2 + s1; });

		for (auto&& au : s) {
			r+=au;
		}

		while (r[0] == '0' && r.size() > 1) {
			r.erase(0,1);
		}

		return r;
	}

	string largestNumber_0(vector<int>& nums) {
		string r;
		vector<vector<int>> nums_v;
		vector<vector<int>> sort_v;

		if (nums.size() == 1) {
			r += to_string(nums[0]);
			return r;
		}

		for (auto au : nums) {
			string temp_s = to_string(au);
			vector<int> temp_v;
			for (int j = 0; j < temp_s.size(); j++) {
				temp_v.emplace_back(temp_s[j] - '0');
			}
			nums_v.emplace_back(temp_v);
		}

		sort_v.emplace_back(nums_v[0]);

		for (int i = 1; i < nums_v.size(); i++) {
			bool is_big = false;

			for (int j = 0; j < sort_v.size(); j++) {
				
				int n = nums_v[i].size();
				int m = sort_v[j].size();
				int nm = n+m;

				vector<int> nums_sort(nums_v[i].begin(), nums_v[i].end());
				vector<int> sort_nums(sort_v[j].begin(), sort_v[j].end());

				nums_sort.insert(nums_sort.end(), sort_v[j].begin(), sort_v[j].end());
				sort_nums.insert(sort_nums.end(), nums_v[i].begin(), nums_v[i].end());

				for (int k = 0; k < nm; k++) {
					if (nums_sort[k] > sort_nums[k]) {
						sort_v.insert(sort_v.begin() + j, nums_v[i]);
						is_big = true;
						break;
					}
					else if (nums_sort[k] < sort_nums[k]) {
						is_big = false;
						break;
					}
				}

				if (is_big) {
					break;
				}
			}

			if (!is_big) {
				sort_v.emplace_back(nums_v[i]);
			}
		}

		int i = 0;
		while (!sort_v[i].empty() && sort_v[i][0] == 0) {
			sort_v[i].erase(sort_v[i].begin());

			if (sort_v[i].empty()) {
				sort_v.erase(sort_v.begin());
			}

			if (sort_v.empty()) {
				r = "0";
				return r;
			}
		}

		for (int i = 0; i < sort_v.size(); i++) {
			for (int j = 0; j < sort_v[i].size(); j++) {
				r += to_string(sort_v[i][j]);
			}
		}

		return r;
	}

	/*=====================================================================*/

	/*=================     50. Pow(x, n) 220616    =======================*/
	double myPow(double x, int n) {
		double r = 0;

		if (x == 0) {
			return x;
		}
		else if (n == 0) {
			return 1;
		}
		else if (x == 1 || n == 1) {
			return x;
		}
		else if ((x > 0 && x < 0.0001) || (x < 0 && x > -0.0001)) {
			return 0;
		}
		else if (x != 1 && x != -1 && n == -2147483648) {
			return 0;
		}
		else if (x == -1 && n % 2 == 0 ) {
			return 1;
		}
		else if (x == -1 && n % 2 != 0) {
			return -1;
		}

		if (n < 0) {
			x = 1 / x;
			n *= -1;
		}

		r = recursive_50(x, n);

		return r;
	}

	double recursive_50(double x, int n) {
		if (n == 0) {
			return 1;
		}
		else if (n == 1) {
			return x;
		}

		double temp_r = x * recursive_50(x, n - 1);

		if ((x > 0 && x < 0.0001) || (x < 0 && x > -0.0001)) {
			return 0;
		}

		return temp_r;
	}
	/*=====================================================================*/

	/*==============     324. Wiggle Sort II 220620    ====================*/

	void wiggleSort(vector<int>& nums) {

		int n = nums.size();

		if (n == 1) {
			return;
		}
		else if (n == 2) {
			sort(nums.begin(),nums.end());
			return;
		}

		//nth_element(nums.begin(), nums.begin()+4, nums.end());

		int half = n / 2;

		sort(nums.begin(),nums.end(), greater());

		for (int i = 0; i < n; i+=2) {
			nums.insert(nums.begin() + i, nums.back());
			nums.pop_back();
		}

		int j = 0;
		if (n % 2 == 0) {//even
			for (int i = n - 1; i > 0; i -= 2) {
				if (nums[i - 1] == nums[i]) {
					if (i - 2 >= 0 && nums[i - 2] < nums[i - 1]) {
						swap(nums[j], nums[i]);
						j += 2;
						continue;
					}
					else {
						swap(nums[j], nums[i - 1]);
						j += 2;
					}
				}
				else if (nums[i - 1] > nums[i]) {
					swap(nums[i - 1], nums[i]);
				}
			}
		}
		else {// odd
			j = 1;
			for (int i = n - 1; i > 0; i -= 2) {
				if (nums[i - 1] == nums[i]) {
					if (i - 2 >= 0 && nums[i - 2] > nums[i - 1]) {
						swap(nums[j], nums[i]);
						j += 2;
						continue;
					}
					else {
						swap(nums[j], nums[i - 1]);
						j += 2;
					}
				}
				else if (nums[i - 1] < nums[i]) {
					swap(nums[i - 1], nums[i]);
				}
			}
		}

	}

	/*=====================================================================*/

	/*===============    5. Longest Palindromic Substring   ===============*/

	string longestPalindrome(string s) {
		int n = s.size();

		if (n == 1) {
			return s;
		}

		vector<vector<bool>> dp(n, vector<bool>(n, false));

		string res;
		int l = 0;
		int r = 0;

		for (int i = 0; i < n-1; i++) {
			int j = i;

			while (j + 1 < n && s[i] == s[j+1]) {
				j++;
			}
			
			if (s[i] == s[j] && j - i > r - l) {
				l = i;
				r = j;
			}

			int ii = i;
			int jj = j;

			while (ii-1 >= 0 && jj + 1 < n && s[ii-1] == s[jj+1]) {
				ii--;
				jj++;
			}

			if (jj - ii > r - l) {
				l = ii;
				r = jj;
			}
		}

		res = s.substr(l,r-l+1);
		return res;
	}

	string longestPalindrome_0(string s) {
		int n = s.size();

		if (n == 1) {
			return s;
		}

		vector<vector<bool>> dp(n, vector<bool>(n, false));

		string res;
		int l = 0;
		int r = 0;

		for (int i = 0; i < n; i++) {
			dp[i][i] = true;
			if (i + 1 < n) {
				dp[i][i+1] = (s[i] == s[i + 1]);

				if (dp[i][i + 1]) {
					l = i;
					r = i + 1;
				}
				
			}
		}

		for (int j = 2; j <n ; j++) {
			for (int i = j - 2; i >=0 ; i--) {
				dp[i][j] = (dp[i+1][j-1] && s[i] == s[j]);

				if (dp[i][j] && j - i > r - l) {
					r = j;
					l = i;
				}
			}
		}

		/*for (int i = n-3; i >=0 ; i--) {
			for (int j = i + 2; j < n; j++) {
				dp[i][j] = (dp[i+1][j-1] && s[i] == s[j]);

				if (dp[i][j] && j - i > r - l) {
					r = j;
					l = i;
				}
			}
		}*/

		res = s.substr(l, r - l + 1);
		return res;
	}

	/*=====================================================================*/

	/*=========================       15. 3Sum      =======================*/

	vector<vector<int>> threeSum(vector<int>& nums) {

		int n = nums.size();
		vector<vector<int>> res;

		if (n < 3) {
			return {};
		}
		else if (nums[0] > 0) {
			return {};
		}
		else if (n == 3) {
			if (nums[0] + nums[1] + nums[2]==0) {
				res.emplace_back(nums);
			}
			return res;
		}

		sort(nums.begin(), nums.end());

		for (int i = 0; i < n - 2; i++) {

			if (nums[i] > 0) {     //If number fixed is +ve, stop there because we can't make it zero by searching after it.
				break;
			}

			if (i > 0 && nums[i] == nums[i - 1]) {
				continue;
			}

			int l = i + 1;
			int r = n - 1;

			while (l < r) {
				if (nums[i] + nums[l] + nums[r] < 0) {
					l++;
				}
				else if (nums[i] + nums[l] + nums[r] > 0) {
					r--;
				}
				else {

					res.emplace_back(vector<int>{ nums[i], nums[l], nums[r] });

					while (l + 1 < r && nums[l] == nums[l + 1]) {
						l++;
					}

					while (r - 1 > l && nums[r] == nums[r -1]) {
						r--;
					}

					l++;
					r--;
				}
			}

		}

		return res;
	}


	vector<vector<int>> threeSum_0(vector<int>& nums) {

		int n = nums.size();
		vector<vector<int>> res;
		map<int, int> mp;

		sort(nums.begin(), nums.end());

		if (n < 3) {
			return {};
		}
		else if (nums[0] > 0) {
			return {};
		}
		else if (n == 3) {
			if (nums[0] + nums[1] + nums[2] == 0) {
				res.emplace_back(nums);
			}
			return res;
		}

		for (int i = 0; i < n; i++) {
			mp[nums[i]] = i;
		}

		for (int i = 0; i < n - 2; i++) {

			if (nums[i] > 0) {     //If number fixed is +ve, stop there because we can't make it zero by searching after it.
				break;
			}

			for (int j = i + 1; j < n - 1; j++) {
				
				int target = -(nums[i] + nums[j]);

				if (mp[target] > 0 && mp[target] > j) {
					res.emplace_back(vector<int>{ nums[i], nums[j], nums[mp[target]] });
				}
				
				j = mp[nums[j]];
			}

			i = mp[nums[i]];
		}

		return res;
	}

	/*=====================================================================*/


	/*=============   98. Validate Binary Search Tree   ===================*/

	bool isValidBST(TreeNode* root) {
		bool r = true;

		postorder_98(root, r);

		return r;
	}

	vector<int> postorder_98(TreeNode* node, bool& isb) {
		if (!isb) return {};

		if (node->left && node->right) {
			vector<int> ll = postorder_98(node->left, isb);
			vector<int> rr = postorder_98(node->right, isb);

			for (auto au : ll) {
				if (node->val <= au) {
					isb = false;
				}
			}

			for (auto au : rr) {
				if (node->val >= au) {
					isb = false;
				}
			}

			ll.emplace_back(node->val);
			ll.insert(ll.end(), rr.begin(), rr.end());

			return ll;
		}
		else if (node->left) {
			vector<int> ll = postorder_98(node->left, isb);

			for (auto au : ll) {
				if (node->val <= au) {
					isb = false;
				}
			}

			ll.emplace_back(node->val);

			return ll;
		}
		else if (node->right) {
			vector<int> rr = postorder_98(node->right, isb);

			for (auto au : rr) {
				if (node->val >= au) {
					isb = false;
				}
			}
			rr.emplace_back(node->val);
			
			return rr;
		}
		else {
			return vector<int>{node->val};
		}
	}

	/*=====================================================================*/

	/*======================   91. Decode Ways 220624 15:14 ==========================*/

	int numDecodings(string s) {

		if (s[0] == '0') return 0;

		int n = s.size();
		
		int res = 1;
		int prev = 0;

		for (int i = n - 1; i >= 0; i--) {
			int temp_nums = 0;

			if (s[i] != '0') {
				temp_nums = res;
			}
			
			if (i < n - 1 && (s[i] == '1' ||(s[i] == '2' && s[i+1] < '7'))) {
				temp_nums += prev;
			}
			prev = res;
			res = temp_nums;

		}

		return res;
	}

	int recursive_91(string s, int od, vector<int>& nums) {
		if (od == -1) {
			od = 0;
		}

		if (nums[od] >= 0) {
			return nums[od];
		}

		int num = s[od] - '0';
		int prev = s[od - 1] - '0';
		string sum_str = s.substr(od - 1, 2);
		int sum = stoi(sum_str);
		int res = 0;

		if (num != 0) {
			res = recursive_91(s, od - 1, nums);

			if (prev != 0 && sum < 27) {
				res += recursive_91(s, od - 2, nums);
			}
		}
		else if(sum < 27) {
			if (prev == 0) return nums[od] = 0;

			res = recursive_91(s, od - 2, nums);
		}

		return nums[od] = res;
	}

	/*===========================================================================*/

	/*======================   338. Counting Bits 220626 20：13 ==========================*/
	vector<int> countBits(int n) {

		/*if (n == 0) {
			return vector<int>{0, 1};
		}*/

		vector<int> dp(n + 1, 0);
		//dp[1] = 1;

		for (int i = 0; i <= n; i++) {
			if (i % 2) {
				dp[i] = dp[i / 2] + 1;
			}
			else {
				dp[i] = dp[i / 2];
			}
		}

		return dp;
	}
	/*===========================================================================*/

	/*=====================   647. Palindromic Substrings 10:18  ==========================*/
	int countSubstrings(string s) {

	}
	/*=====================================================================================*/

	/*=====================     226. Invert Binary Tree 10:29    ==========================*/
	TreeNode* invertTree(TreeNode* root) {

		vector<int> node_val;

		TreeNode* root_temp = root;
		//preorder_traverse_226(root_temp, node_val);

		//if (!(node_val.size() & 1)) {//even
		//	node_val.emplace_back(INT_MIN);
		//}

		//root_temp = root;

		//vector<Solution::TreeNode> node_vector_226(node_val.rbegin(), node_val.rend());
		//std::reverse(node_val.begin(), node_val.end());
		//TreeNode* root_new_temp = new TreeNode(INT_MIN);
		//root = root_new_temp;

		//preorder_traverse_build_226(root_new_temp, node_val);
		preorder_traverse_swap_226(root_temp);

		return root;
	}

	void preorder_traverse_swap_226(TreeNode* root) {
		if (!root) return;

		TreeNode* temp = root->left;

		root->left = root->right;
		root->right = temp;

		preorder_traverse_swap_226(root->left);
		preorder_traverse_swap_226(root->right);

		return;
	}

	void preorder_traverse_226(TreeNode* root, vector<int>& node_val) {
		if (!root) return;

		if (root->left)
			preorder_traverse_226(root->left, node_val);
		else if(root->right) {
			node_val.emplace_back(INT_MIN);
		}

		node_val.emplace_back(root->val);

		if (root->right)
			preorder_traverse_226(root->right, node_val);
		else if (root->left) {
			node_val.emplace_back(INT_MIN);
		}
	}

	

	TreeNode* preorder_traverse_build_226(TreeNode* root, vector<int>& node_val) {
		/*if (node_val.back() == INT_MIN) {
			node_val.pop_back();
			return root;
		}*/

		if (node_val.empty()) {
			return root;
		}

		//if (node_val.back() != INT_MIN){
			root -> left = new TreeNode(node_val.back());
			//node_val.pop_back();
			//preorder_traverse_build_226(root->left, node_val);
		//}
		node_val.pop_back();

		//root = new TreeNode(node_val.back(), root->left, nullptr);
		root->val = node_val.back();
		node_val.pop_back();

		//if (node_val.back() != INT_MIN) {
			root->right = new TreeNode(node_val.back());
			//node_val.pop_back();
			//preorder_traverse_build_226(root->right, node_val);
		//}

		return root;
	}
	/*=====================================================================================*/

	/*================     543. Diameter of Binary Tree 220628 10：25    ==================*/

	int diameterOfBinaryTree(TreeNode* root) {
		int n = 0;
		vector<int> single_path;
		vector<vector<int>> path;

		TreeNode* temp_root = root;
		n = dfs_543(temp_root, n);

		return n;
	}

	int dfs_543(TreeNode* root, int& n) {
		
		if (!root) {
			return 0;
		}

		int ln = dfs_543(root->left, n);

		int rn = dfs_543(root->right, n);

		n = max(n, ln + rn);

		return max(ln, rn) + 1;
	}

	/*=====================================================================================*/


};
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/









/*&&&&&&&&&&&&&&&&&&&&&&&&&&&         Class MyLinkedList        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

class MyLinkedList {
public:
	struct Node {
		int val = NULL;
		Node* next = nullptr;
		Node(int val) :val(val), next(nullptr) {}
	};

	Node* head = new Node(NULL);
	int size = NULL;
	/** Initialize your data structure here. */
	MyLinkedList() {
		//head = new Node(NULL);

		size = 0;
	}

	/** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
	int get(int index) {
		if (index >= size || index < 0) return -1;
		Node* temp = head->next;
		for (int i = 0; i < index; i++) {
			temp = temp->next;
		}
		return temp->val;
	}

	/** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
	void addAtHead(int val) {
		Node* temp = head->next;
		head->next = new Node(val);
		head->next->next = temp;
		size++;
	}

	/** Append a node of value val to the last element of the linked list. */
	void addAtTail(int val) {
		Node* temp = head;
		while (temp->next) temp = temp->next;
		temp->next = new Node(val);
		size++;
	}

	/** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
	void addAtIndex(int index, int val) {
		if (index > size) return;
		Node* temp = head;
		for (int i = 0; i < index; i++) {
			temp = temp->next;
		}
		Node* temp_node = temp->next;
		temp->next = new Node(val);
		temp->next->next = temp_node;
		size++;
	}

	/** Delete the index-th node in the linked list, if the index is valid. */
	void deleteAtIndex(int index) {
		if (index >= size) return;
		Node* temp = head;
		for (int i = 0; i < index; i++) temp = temp->next;
		Node* temp_del = temp->next;
		temp->next = temp_del->next;
		size--;
		delete temp_del;
		temp_del->next = nullptr;
	}
};
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&        146. Class LRUCache        &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
class LRUCache {
public:

	int n = 0;

	list<pair<int, int>> l;
	unordered_map<int, list<pair<int, int>>::iterator> m;

	LRUCache(int capacity) {
		n = capacity;
	}

	int get(int key) {
		if (m.find(key) == m.end())
			return -1;
		l.splice(l.begin(), l, m[key]);
		return m[key]->second;
	}

	void put(int key, int value) {

		if (m.find(key) != m.end())
		{
			l.splice(l.begin(), l, m[key]);
			m[key]->second = value;
			return;
		}

		if (l.size() == n)
		{
			auto d_key = l.back().first;
			l.pop_back();
			m.erase(d_key);
		}
		l.push_front({ key,value });
		m[key] = l.begin();
	}

};
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/


/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&     1476. Class Subrectangle Queries    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

class SubrectangleQueries {
public:

	vector<vector<int>> _rectangle;
	int _row = 0;
	int _col = 0;

	SubrectangleQueries(vector<vector<int>>& rectangle) {

		_row = rectangle.size();
		_col = rectangle[0].size();

		_rectangle.assign(_row, vector<int>(_col, 0));
		copy_n(rectangle.begin(), rectangle.size(), _rectangle.begin());

	}

	void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
		for (int i = row1; i < row2; i++) {
			for (int j = col1; j < col2; j++) {
				_rectangle[i][j] = newValue;
			}
		}
	}

	int getValue(int row, int col) {
		return _rectangle[row][col];
	}
};


/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList obj = new MyLinkedList();
 * int param_1 = obj.get(index);
 * obj.addAtHead(val);
 * obj.addAtTail(val);
 * obj.addAtIndex(index,val);
 * obj.deleteAtIndex(index);
 */

 //220623 build a binary search tree (BST) from level-first order.
template <typename T>
T* Solution::build_b_tree_level_order(vector<T>& node_vector, T* root) {

	root = &node_vector[0];

	const int n = node_vector.size();
	queue<TreeNode*> q;

	q.emplace(&node_vector[0]);
	int i = 0;

	while (!q.empty() && i < n) {

		Solution::TreeNode* temp_node = q.front();
		q.pop();

		i++;
		if (i < n && node_vector[i].val != INT_MIN) {
			temp_node->left = &node_vector[i];
			q.emplace(temp_node->left);
		}

		i++;
		if (i < n && node_vector[i].val != INT_MIN) {
			temp_node->right = &node_vector[i];
			q.emplace(temp_node->right);
		}
	}

	return root;
}

vector<int> Solution::twoSum(vector<int>& nums, int target) {
	map <int, pair<int, int>> mp;
	int i;
	vector <int> v;
	for (i = 0; i < nums.size(); i++)
		mp[nums[i]] = make_pair(1, i);

	for (i = 0; i < nums.size(); i++)
	{
		if (mp[target - nums[i]].first > 0 && mp[target - nums[i]].second != i)
		{
			v.push_back(i), v.push_back(mp[target - nums[i]].second);
			return v;
		}
	}
}

int Solution::reverse(int x) {
	long long res = 0;
	while (x) {
		res = res * 10 + x % 10;
		x /= 10;
	}
	return (res<INT_MIN || res>INT_MAX) ? 0 : res;
}

bool Solution::isPalindrome(int x) {
	if (x < 0 || (x != 0 && x % 10 == 0)) return false;
	int sum = 0;
	while (x > sum) {
		sum = sum * 10 + x % 10;
		x /= 10;
	}
	return x == sum || x == sum / 10;
}

int Solution::romanToInt(string s) {
	map<char, int> T = { {'I',1}, { 'V' , 5 }, { 'X' , 10 }, { 'L' , 50 }, { 'C' , 100 },{ 'D' , 500 }, { 'M' , 1000 } };
	int sum = T[s.back()];

	for (int i = s.length(); i >= 0; i++) {
		if (T[s[i]] < T[s[i + 1]])
			sum -= T[s[i]];
		else
			sum += T[s[i]];
	}

	return sum;
}

string Solution::longestCommonPrefix(vector<string>& strs) {
	string prefix = "";
	for (int idx = 0; strs.size() > 0; prefix += strs[0][idx], idx++)
		for (int i = 0; i < strs.size(); i++)
			if (idx >= strs[i].size() || (i > 0 && strs[i][idx] != strs[i - 1][idx]))
				return prefix;
	return prefix;
}

//190105
bool Solution::isValid(string s) {
	stack<char> stk;
	for (auto&& c : s)
		switch (c) {
		case'(': stk.push(')'); break;
		case'[': stk.push(']'); break;
		case'{': stk.push('}'); break;
		default:
			if (stk.empty() || c != stk.top()) return false;
			else stk.pop();
		}
	return stk.empty();
}



Solution::ListNode* Solution::mergeTwoLists(Solution::ListNode* l1, Solution::ListNode* l2) {
	Solution::ListNode header(LONG_MIN);
	Solution::ListNode* tail_ptr = &header;

	while (l1 && l2) {
		Solution::ListNode** next_node = (l1->val < l2->val ? &l1 : &l2);
		tail_ptr->next = *next_node;
		*next_node = (*next_node)->next;
		tail_ptr = tail_ptr->next;
	}
	tail_ptr->next = l1 ? l1 : l2;
	return header.next;
}

//190107
int Solution::removeDuplicates(vector<int>& nums) {
	//auto* first_ptr=nums
	//auto count = 0;

	//if (nums.empty()) return count;
	//for (int i = 1; i < nums.size();i++) {
	//	if (nums[i] == nums[i-1]){
	//		count++;
	//	}
	//	else{
	//		nums[i-count]=nums[i];
	//		//++it;
	//	}
	//		
	//}

	//return nums.size()-count;
	nums.erase(unique(nums.begin(), nums.end()), nums.end());
	return nums.size();
}

//190108
int Solution::removeElement(vector<int>& nums, int val) {
	/*int length = nums.size();
	int i = 0;
	while(i<length) {
		nums[i] == val ? nums[i] = nums[--length] : i++;
	}

	return length;*/

	while (find(nums.begin(), nums.end(), val) != nums.end())
		nums.erase(find(nums.begin(), nums.end(), val));
	return nums.size();
}

//190114
bool Solution::checkPossibility(vector<int>& nums) {
	//bool modified = false;
	//for (int i = 1; i < nums.size();i++) {
	//	if (nums[i - 1] > nums[i]) {
	//		if (modified++) return false;
	//		i - 2 < 0 || nums[i - 2] <= nums[i] ? nums[i - 1] = nums[i] : nums[i] = nums[i - 1];
	//	}

	//}
	return true;
}

//190115
int Solution::largestPalindrome(int n) {
	if (n == 1)return 9;
	long long max = pow(10, n) - 1;
	for (int v = max - 1; v > (max / 10); v--) {
		string s = to_string(v), s0 = s;
		std::reverse(s.begin(), s.end());
		long long u = atoll((s0 + s).c_str());
		//long long u = stoll(s0 + s);
		for (long long x = max; x * x >= u; x--)if (u % x == 0)return(int)(u % 1337);
	}
	return 0;
}

//190116
bool Solution::buddyStrings(string A, string B) {
	if (A.size() == 1 || A.size() != B.size()) return false;

	int numChar_A = unordered_set<char>(A.begin(), A.end()).size();
	if (A == B && numChar_A < B.size()) return true;
	int i = 0, N = (int)A.size(), j = N - 1;
	while (i < N - 1 && A[i] == B[i]) ++i;
	while (j > 0 && A[j] == B[j]) --j;
	swap(A[i], A[j]);
	return A == B;

}

//190117
int Solution::countPrimes(int n) {
	if (n <= 2) return 0;
	vector<bool> Primes(n, true);
	int sum = 1;
	int low = sqrt(n);
	for (int i = 3; i < n; i += 2) {
		if (Primes[i]) {
			sum++;
			if (i > low) continue;
			for (int j = i * i; j < n; j += i * 2) {
				Primes[j] = false;
			}
		}
	}

	return sum;
}

//190118
void Solution::rotate(vector<int>& nums, int k) {
	k %= nums.size();
	std::reverse(nums.begin(), nums.end());
	std::reverse(nums.begin(), nums.begin() + k);
	std::reverse(nums.begin() + k, nums.end());
	//2
	std::rotate(nums.begin(), nums.end() - k, nums.end());

}

//190121
string Solution::convertToTitle(int n) {
	return n == 0 ? "" : convertToTitle((n - 1) / 26) + (char)((n - 1) % 26 + 'A');
}

//190122
bool Solution::isBadVersion(int n) {
	return true;
}

int Solution::firstBadVersionRec(int L, int R) {
	if (L == R) return L;
	else {
		int Middle = L + (R - L) / 2;
		if (isBadVersion(Middle)) return firstBadVersionRec(L, Middle);
		else return firstBadVersionRec(Middle + 1, R);

	}
}

//190221
int Solution::findPairs(vector<int>& nums, int k) {
	if (k < 0) return 0;
	int count = 0;
	unordered_multiset<int> ums(nums.begin(), nums.end());
	unordered_set<int> us(nums.begin(), nums.end());
	for (auto i : us) {
		if (ums.count(i + k) > !k) count++;
	}
	return count;
}

//190222
int Solution::findUnsortedSubarray(vector<int>& nums) {
	int n = nums.size();

	vector<int> maxlhs(n);  //max number from left to cur
	vector<int> minrhs(n);  //min number from right to cur

	for (int i = n - 1, minr = INT_MAX; i >= 0; i--)
		minrhs[i] = minr = min(minr, nums[i]);

	for (int i = 0, maxl = INT_MIN; i < n; i++)
		maxlhs[i] = maxl = max(maxl, nums[i]);

	int i = 0, j = n - 1;

	while (i < n && nums[i] <= minrhs[i]) i++;
	while (j > i && nums[j] >= maxlhs[j]) j--;

	return j - i + 1;
}

//190223
int Solution::findNthDigit(int n) {
	int base = 9;
	long digits = 1;

	while (n - base * digits > 0) {
		n -= base * digits;
		base *= 10;
		digits++;
	}

	int index = (n - 1) % digits;
	int offset = (n - 1) / digits;
	long start = pow(10, digits - 1);
	return to_string(start + offset)[index] - '0';
}

//190224
uint32_t Solution::reverseBits(uint32_t n) {
	string str = bitset<32>(n).to_string();
	std::reverse(str.begin(), str.end());
	return (bitset<32>(str).to_ullong());
}


//190227
bool Solution::isPalindrome(string s) {
	if (s.size() <= 1) return true;

	auto left_end = s.begin();
	auto right_end = s.end();

	while (left_end < right_end) {
		while (!isalnum(*left_end) && left_end < s.end()) left_end++;
		while (!isalnum(*right_end) && right_end > s.begin()) right_end--;
		if (left_end > right_end) return true;
		if (tolower(*left_end) != tolower(*right_end)) return false;
		left_end++;
		right_end--;
	}
	return true;
}

//190303
int Solution::mySqrt(int x) {
	long val = x;
	while (val * val > x) {
		val = (val + x / val) / 2;
	}

	return val;
}

//190304
bool Solution::canPlaceFlowers(vector<int>& flowerbed, int n) {
	flowerbed.insert(flowerbed.begin(), 0);
	flowerbed.push_back(0);

	for (int i = 1; i < flowerbed.size() - 1; ++i) {
		if (flowerbed[i - 1] + flowerbed[i] + flowerbed[i + 1] == 0) {
			--n;
			++i;
		}
	}
	return n <= 0;
}

////190310
//int Solution::findRadius(vector<int>& houses, vector<int>& heaters) {
//	sort(houses.begin(),houses.end());
//	sort(heaters.begin(),heaters.end());
//	vector<int> res(houses.size(), INT_MAX);
//
//	for (int i = 0, h = 0; i < houses.size() && h < heaters.size();) {
//		if (houses[i] <= heaters[h]) {
//			res[i] = heaters[h] - houses[i];
//			i++;
//
//		}
//		else {
//			h++;
//		}
//	}
//
//	for (int i = houses.size() - 1, h = heaters.size() - 1; i >= 0 && h >= 0;) {
//		if (houses[i] >= heaters[h]) {
//			res[i] = min(res[i], houses[i] - heaters[h]);
//			i--;
//		}
//		else {
//			h--;
//		}
//	}
//
//	return *max_element(res.begin(),res.end());
//}
//
////190316
//int Solution::repeatedStringMatch(string A, string B) {
//	// auto searcher = boyer_moore_searcher(begin(B), end(B));
//	auto searcher = 0;// boyer_moore_horspool_searcher(begin(B), end(B));
//
//	auto N = 1;
//	auto T = A;
//
//	auto iter = search(begin(A), end(A), searcher);
//	while (iter == end(A))
//	{
//		A.append(T);
//		N++;
//
//		iter = search(begin(A), end(A), searcher);
//
//		if (B.size() + T.size() <= A.size())
//			break;
//	}
//
//	return (iter != end(A)) ? N : -1;
//}

//220222
bool Solution::isPowerOfThree(int n) {

	if (n == 0) return false;
	if (n == 1) return true;

	while (n % 3 == 0) {
		int temp = n % 10;
		switch (temp) {
		case 1:
		case 3:
		case 7:
		case 9: {
			//int sum = 0;
			//while( n != 0){
				//sum += n % 10;
				//n /= 10;
			//}

			//while(abs(sum) != 1 && sum % 3 == 0){
				//sum /= 3;
			//}

			//if(sum % 3 == 0) return true;

			n /= 3;

			//return false;
		}
		default:
			return false;
		}
	}

	if (n == 1) return true;
	else return false;
}

//220225
int Solution::strStr(string haystack, string needle) {

	/*if (haystack.size() == 0 && needle.size() == 0) return 0;
	if (haystack.size() == 0) return -1;
	if (needle.size() == 0) return 0;
	if (haystack.size() < needle.size()) return -1;

	if (haystack.size() == needle.size()) {
		int count = 0;
		for (int i = 0; i < haystack.size(); i++) {
			if (haystack[i] == needle[i]) {
				count++;
				if (haystack[i] != needle[i]) {
					return -1;
				}
			}
		}

		if (count == haystack.size()) {
			return  0;
		}
	}

	for (int i = 0; i < haystack.size() - needle.size() + 1; i++) {

		if (haystack[i] == needle[0]) {
			int position = i;
			int i_tmep = i;
			int j = 1;
			for (; j < needle.size(); j++) {
				i_tmep++;
				if (haystack[i_tmep] != needle[j]) {
					break;
				}
			}

			if (j == needle.size()) return position;
		}
	}

	return -1;*/

	return 0;
}

//vector<int> Solution::find_KMP(const string& const needle) {
	//vector<int> lsp;
	//int len = 0;
	//int i = 0;
	/*while (i < needle.size()) {
		if (needle[i] == needle[len]) {
			vector.empalce_back(i);
			i++;
			len++;
		}
		else if (len > 0) {
			len = needle[len--];
		}
	}*/
	//}

string Solution::countAndSay(int n) {
	string temp = "1";
	if (n == 1) return temp;
	string result;
	int i = 1;
	while (i < n) {
		int j = 0;
		while (j < temp.size()) {
			int count = 1;
			while (j + 1 < temp.size() && temp[j] == temp[j + 1]) {
				count++;
				j++;
			}
			result += std::to_string(count) + temp[j];
			j++;
		}
		temp = result;
		result = "";
		i++;
	}

	return temp;

	//return count_and_say(1, n, result);
}

string Solution::count_and_say(int i, int& n, string temp) {
	i++;
	string result;

	if (temp.size() == 1) {
		result += "1";
		result += temp[0];
		return count_and_say(i, n, result);
	}

	for (int j = 0; j < temp.size(); j++) {
		int count = 1;
		while (j + 1 < temp.size() && temp[j] == temp[j + 1]) {
			count++;
			j++;
		}
		result += std::to_string(count);
		result += temp[j];
	}

	if (i == n) return result;

	return count_and_say(i, n, result);
}

//220413
void Solution::setZeroes(vector<vector<int>>& matrix) {
	int n = matrix.size();
	int m = matrix[0].size();

	bool row0 = false;
	bool col0 = false;

	for (int i = 0; i < n; i++) { if (matrix[i][0] == 0) col0 = true; }

	for (int j = 0; j < m; j++) { if (matrix[0][j] == 0) row0 = true; }

	for (int i = 1; i < n; i++) {
		for (int j = 1; j < m; j++) {
			if (matrix[i][j] == 0) {
				matrix[i][0] = matrix[0][j] = 0;
			}
		}
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			if (matrix[i][0] == 0 || matrix[0][j] == 0) {
				matrix[i][j] = 0;
			}

		}

	}

	for (int i = 0; i < n; i++) {
		if (col0) matrix[i][0] = 0;
	}

	for (int j = 0; j < m; j++) {
		if (row0) matrix[0][j] = 0;
	}
}

struct Solution::Node138 {
public:
	int val;
	Node138* next;
	Node138* random;

	Node138(int _val) {
		val = _val;
		next = NULL;
		random = NULL;
	}
};

Solution::Node138* Solution::copyRandomList(Node138* head) {
	if (!head) return nullptr;
	Node138* temp = head;

	if (!head->next) {
		return temp;
	}

	while (temp) {
		Node138* ne = new Node138(0);
		ne->next = temp->next;
		ne->val = temp->val;
		temp->next = ne;
		//if(temp->random->next) ne->random
		temp = ne->next;
	}

	temp = head;
	Node138* new_head = head->next;

	while (temp && temp->next) {
		if (temp->random) {
			temp->next->random = temp->random->next;
		}
		else {
			temp->next->random = nullptr;
		}
		temp = temp->next->next;
	}

	temp = head->next;
	Node138* temp_o = head;

	while (temp->next) {
		temp_o->next = temp_o->next->next;
		temp_o = temp_o->next;

		if (!temp_o->next->next) temp_o = nullptr;

		temp->next = temp->next->next;
		temp = temp->next;
	}

	temp = head;
	while (temp) {

		if (!temp->next->next) {
			temp->next = nullptr;
			break;
		}

		temp = temp->next;
	}



	return new_head;
}

//void Solution::GetResult(int* p, int& Get_Result) {
//
//	//int a[6][2] = { {1000,5}, {800,2}, {400,5}, {300,5}, {400,3} , {200,2} };
//	//p = a[0];
//	int N = p[0];
//	int n = p[1];
//	vector<vector<int>> v;
//	for (int i = 2; i < n*2+2; i+=2) {
//		vector<int> temp = { p[i], p[i + 1] };
//		v.emplace_back(temp);
//	}
//	cout << v[0][0] << endl;
//	//get_result(v, Get_Result);
//}
//
//void Solution::get_result(vector<vector<int>>& v, int& sum_m, int sum, int& Get_Result) {
//	for () {
//
//	}
//}

int Solution::huawei0() {
	/*int N = 1000;
	int m = 3;
	vector<vector<int>> bag;
	vector<vector<int>>dp(m + 1,vector<int>(N + 1, 0));
	vector<vector<int>> price = { {0,0,0}, {800,400,300}, {400,0,0},  {500,0,0} };
	vector<vector<int>> priority = { {0,0,0}, {2,5,5}, {3,0,0},  {2,0,0} };*/

	int N;
	int m;
	cin >> N >> m;
	//vector<vector<int>> bag;
	vector<vector<int>>dp(m + 1, vector<int>(N + 1, 0));
	vector<vector<int>> price(61, vector<int>(3, 0));
	vector<vector<int>> priority(61, vector<int>(3, 0));

	int a, b, c;

	for (int i = 1; i <= m; i++) {
		cin >> a >> b >> c;
		if (c == 0) {
			price[i][0] = a;
			priority[i][0] = b;
		}
		else if (price[c][1] == 0) {
			price[c][1] = a;
			priority[c][1] = b;
		}
		else {
			price[c][2] = a;
			priority[c][2] = b;
		}
	}

	for (int i = 1; i <= m; i++) {
		for (int j = 1; j <= N; j++) {

			int a = price[i][0], b = priority[i][0];
			int c = price[i][1], d = priority[i][1];
			int e = price[i][2], f = priority[i][2];

			if (j >= a) {
				dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - a] + a * b);
			}
			else {
				dp[i][j] = dp[i - 1][j];
			}

			if (j >= (a + c)) {
				dp[i][j] = max(dp[i][j], dp[i - 1][j - a - c] + a * b + c * d);
			}
			else {
				dp[i][j] = dp[i][j];
			}

			if (j >= (a + e)) {
				dp[i][j] = max(dp[i][j], dp[i - 1][j - a - e] + a * b + e * f);
			}
			else {
				dp[i][j] = dp[i][j];
			}

			if (j >= (a + c + e)) {
				dp[i][j] = max(dp[i][j], dp[i - 1][j - a - c - e] + a * b + c * d + e * f);
			}
			else {
				dp[i][j] = dp[i][j];
			}
		}
	}
	int result = dp[m][N];
	cout << result << endl;
	return dp[m][N];
}

int Solution::trap(vector<int>& height) {
	int n = height.size();

	if (n == 1 || n == 2) return 0;

	int i = 0, j = 1;
	int result = 0;

	int ans = 0, current = 0;
	stack<int> st;
	while (current < height.size()) {
		while (!st.empty() && height[current] > height[st.top()]) {
			int top = st.top();
			st.pop();
			if (st.empty())
				break;
			int distance = current - st.top() - 1;
			int bounded_height = min(height[current], height[st.top()]) - height[top];
			ans += distance * bounded_height;
		}
		st.push(current++);
	}

	/*stack<int> s;

	vector<int> left_max(n);
	vector<int> right_max(n);

	left_max[0] = height[0];
	for (int i = 1; i < n; i++) {
		left_max[i] = max(left_max[i - 1], height[i]);
	}

	right_max[n - 1] = height[n-1];
	for (int i = n - 1; i > 0; i--) {
		right_max[i - 1] = max(right_max[i], height[i - 1]);
	}

	for (int i = 0; i < n; i++) {
		result += min(left_max[i], right_max[i]) - height[i];
	}*/

	/*while (j < n) {
		if (height[i] > height[j]) {
			if(!s.empty()){
				while (!s.empty() && height[s.top()] < height[i]) {
					s.pop();
				}

				s.emplace(i);
			}
			else {
				s.emplace(i);
			}
		}
		else if (height[i] < height[j] && !s.empty()) {
			recursive(height, s, j, result);
			if(height[s.top()] <= height[s.top()+1]){
				int t = s.top() + 1;
				while(!s.empty() && s.top()+1 < j && height[s.top()] <= height[s.top()+1]){
					t = s.top() + 1;
					s.pop();
				}
				s.emplace(t);
			}
		}
		i++;
		j++;
	}*/

	return ans;
}


void Solution::recursive(vector<int>& height, stack<int> s, int r, int& result) {

	while (height[s.top()] < height[r] && s.size() > 1) {
		s.pop();
	}

	int l = s.top();
	int h = min(height[l], height[r]);
	int temp = 0;
	for (int i = l + 1; i < r; i++) {
		if (h > height[i]) {
			temp += h - height[i];
			height[i] = h;
		}
	}
	result += temp;
}

int Solution::rob(vector<int>& nums) {

	if (nums.size() == 1) return nums.front();

	int n = nums.size();
	int k = 0;
	int result = 0;
	vector<int> dp(n, 0);
	dp[1] = max(nums[0], nums[1]);
	//result = job_recursive(nums, 0, dp);

	int a = 0;
	int b = max(nums[0], nums[1]);
	int c = 0;

	for (int i = 2; i < n; i++) {
		c = max(a + nums[i], b);
		a = b;
		b = c;
	}

	result = c;

	return result;
}

int Solution::job_recursive(vector<int>& nums, int k, vector<int>& dp) {
	if (k >= nums.size()) {
		return 0;
	}

	if (dp[k] > -1) {
		return dp[k];
	}

	return dp[k] = max(job_recursive(nums, k + 2, dp) + nums[k], job_recursive(nums, k + 1, dp));
}

