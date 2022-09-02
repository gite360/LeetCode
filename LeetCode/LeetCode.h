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
		TreeNode* next;
		TreeNode() : val(0), left(nullptr), right(nullptr), next(nullptr) {}
		TreeNode(int x) : val(x), left(nullptr), right(nullptr), next(nullptr) {}
		TreeNode(int x, TreeNode* left, TreeNode* right) : val(x), left(left), right(right) {}

	};

	//220623 build a binary search tree (BST) from level-first order.
	template <typename T>
	T* build_b_tree_level_order(vector<T>& node_vector, T* root);

	//181230
	/*= ================ = 220831 12:06 Longest Common Prefix     ================*/
	vector<int> twoSum_0(vector<int>& nums, int target);

	vector<int> twoSum(vector<int>& numbers, int target) {
		vector<int> dp;
		int n = numbers.size() - 1;

		while (numbers[0] + numbers[n] > target) {
			n--;
		}

		int i = 0;
		int j = n;

		while (i < j) {
			if (numbers[i] + numbers[j] == target) {
				return vector<int>{i + 1, j + 1};;
			}
			else if (numbers[i] + numbers[j] > target) {
				j--;
			}
			else {
				i++;
			}
		}

		return vector<int>{i + 1, j + 1};
	}
	/*======================================================================*/
	//181231
	int reverse(int x);
	//190102
	bool isPalindrome(int x);
	//190103
	int romanToInt(string s);

	/*==================     220831 12:06 Longest Common Prefix     ================*/
	string longestCommonPrefix(vector<string>& strs) {
		string res = divide_and_conquer(strs, 0, strs.size() - 1);
		return res;
	}

	string divide_and_conquer(vector<string>& strs, int i, int j) {
		if (i == j) {
			return strs[i];
		}
		int m = (i + j) / 2;
		string l = divide_and_conquer(strs, i, m);
		string r = divide_and_conquer(strs, m+1, j);
		string s;
		int k = 0;
		
		while (k < l.size()&& k < r.size()) {
			if (l[k] == r[k]) {
				s += l[k++];
				continue;
			}
			break;
		}

		return s;
	}
	/*======================================================================*/

	//190104
	string longestCommonPrefix_0(vector<string>& strs);

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
	vector<int> find_KMP(const string& const needle);
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

	/*==================     220719 39. Daily Temperatures     ================*/
	//220518 739. Daily Temperatures
	vector<int> dailyTemperatures(vector<int>& temperatures) {
		int n = temperatures.size();
		if (n==1) {
			return vector<int>{0};
		}

		vector<int> day_vector(n, 0);
		stack<pair<int,int>> cool_stack;

		for (int i = 0; i < n-1; i++) {
			if (temperatures[i] < temperatures[i + 1]) {
				day_vector[i] = 1;

				while (!cool_stack.empty() && cool_stack.top().first < temperatures[i + 1]) {
						day_vector[cool_stack.top().second] = i + 1 - cool_stack.top().second;
						cool_stack.pop();
				}
			}
			else {
				cool_stack.emplace(temperatures[i],i);
			}
		}

		return day_vector;

		/*size_t n = temperatures.size();
		vector<int> r(n, 0);
		stack<int> s;

		int count = 0;
		for (int i = 0; i < n; i++) {
			while (!s.empty() && temperatures[s.top()] < temperatures[i]) {
				r[s.top()] = i - s.top();
				s.pop();
			}
			s.emplace(i);
		}*/

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
		//return r;
	}

	/*==================     220519 39. Combination Sum     ================*/
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

	/*==============    394. Decode String 22/07/27 08:48     ==========*/

	string decodeString_1(const string& s) {
		int i = 0;
		stack<int> nums;
		stack<string> st;
		string num_str;
		string str;

		for (auto au:s) {
			if (isdigit(au)) {
				num_str += au;
			}
			else if (isalpha(au)) {
				str += au;
			}
			else if (au == '[') {
				nums.emplace(stoi(num_str));
				num_str.clear();
				st.emplace(str);
				str.clear();
			}
			else if (au == ']') {
				int n = nums.top();
				string temp = str;
				while (--n) {
					str += temp;
				}
				str = st.top() + str;
				st.pop();
				nums.pop();
			}
		}

		return str;
	}

	string decodeString_0(const string& s) {
		int i = 0;
		string res = decodeString_0(s, i);
		return res;
	}

	string decodeString_0(const string& s, int& i) {
		string res;
		int n = s.size();
		stack<string> st;
		string num;
		
		while (i < n && s[i] != ']') {
			if (!isdigit(s[i])) {
				res += s[i++];
			}
			else if (isdigit(s[i])) {
				num.clear();
				while (i < n && isdigit(s[i])) {
					num += s[i++];
				}

				string ss = decodeString_0(s, ++i);
				i++;
				
				int numm = stoi(num);
				while (numm -- > 0) {
					res += ss;
				}
			}
		}

		return res;
	}

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

	/*===========  134. Gas Station  =================*/

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

	/*======= 150. Evaluate Reverse Polish Notation  220720  08:29 =======*/
	int evalRPN(vector<string>& tokens) {
		int n = tokens.size();

		if (n == 1) {
			return stoi(tokens.front());
		}

		stack<int> token_stack;

		for (auto au : tokens) {
			if (au != "+" && au != "-" && au != "*" && au != "/") {
				token_stack.emplace(stoi(au));
			}
			else {
				int right = token_stack.top();
				token_stack.pop();
				int left = token_stack.top();
				token_stack.pop();

				if (au == "+") {
					token_stack.emplace(left + right);
				}
				else if (au == "-") {
					token_stack.emplace(left - right);
				}
				else if (au == "*") {
					token_stack.emplace(left * right);
				}
				else if (au == "/") {
					token_stack.emplace(left / right);
				}
			}
		}

		return token_stack.top();


		//int result = 0;
		//map<char, int> operator_map{ {'+', 0}, { '-', 1 }, { '*', 2 }, { '/', 3 } };
		//map<string, int> operator_mmap{ {" + ", 0}, { " - ", 1 }, { " * ", 2 }, { " / ", 3 } };
		//stack<int> digit;
		//stack<char> opera;
		//digit.emplace(stoi(tokens[0]));
		//digit.emplace(stoi(tokens[1]));

		////recursive_150(tokens, operator_map, 0);

		///*---------------------------   Recursive   -------------------------------------------*/
		//int m = n - 1;
		//result = resursive_150(tokens, operator_map, m);
		//return result;
		///*-------------------------------------------------------------------------------------*/

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

		//return digit.top();
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

	/*------   220824 10:22   --------*/
	int lengthOfLongestSubstring(string s) {
		int n = s.size();
		if (s.empty()) return 0;
		else if (n==1) return 1;

		int dp = 1;
		map<char, int> m;
		int start = 0;
		for (int i = 0; i < n; i++) {
			if (!m[s[i]]) {
				m[s[i]] = i+1;
				dp = max(dp, i - start + 1);
			}
			else {
				start = max(start, m[s[i]]);
				m[s[i]] = i+1;
				dp = max(dp, i - start + 1);
			}
		}
		return dp;
	}




	int lengthOfLongestSubstring_0(string s) {
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

		//postorder_98(root, r);

		//TreeNode* root_parent = nullptr;
		//r = inorder_98(root_parent, root);

		r = validate_98(nullptr, root, nullptr);

		return r;
	}

	bool validate_98(TreeNode* left, TreeNode* root, TreeNode* right) {
		if (!root) 
			return true;

		if ((left && root->val <= left->val) || (right && root->val >= right->val)) {
			return false;
		}

		return validate_98(root->left, left,  root) && validate_98(root, root->right, right);
	}

	bool inorder_98(TreeNode*& root_parent, TreeNode* root) {
		if (!root) 
			return true;

		if (!inorder_98(root_parent, root->left)) 
			return false;
		

		if (root_parent && root->val <= root_parent->val) 
			return false;
		
		root_parent = root;
		return inorder_98(root_parent, root->right);
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

	/*================     35. Search Insert Position 220629 09：30    ==================*/

	int searchInsert(vector<int>& nums, int target) {

		int n = nums.size();

		if (target < nums.front()) {
			return 0;
		}
		else if (target > nums.back()) {
			return n;
		}

		int l = 0;
		int r = n - 1;
		int m = 0;

		while (l <= r) {
			m = (l + r) / 2;

			if (l + 1 == r && target > nums[l] && target < nums[r]) {
				return r;
			}
			else if (target < nums[m]) {
				r = m - 1;
				continue;
			}
			else if (target > nums[m]) {
				l = m + 1;
				continue;
			}
			else {
				return m;
			}
		}

		/*if (target > nums[m]) {
			return m + 1;
		}*/

		return l;
	}

	/*=====================================================================================*/

	/*================     Binary Tree Preorder Traversal 220630 11：13    ==================*/
	vector<int> preorderTraversal(TreeNode* root) {
		vector<int> res;
		preorder_travel(root, res);
		return res;
	}

	void preorder_travel(TreeNode* root, vector<int>& res) {
		if (!root) return;
		res.emplace_back(root->val);
		preorder_travel(root->left, res);
		preorder_travel(root->right, res);
		return;
	}
	/*=====================================================================================*/

	/*================     Binary Tree inorderTraversal Traversal 220630 11：13    ==================*/
	vector<int> inorderTraversal(TreeNode* root) {
		vector<int> res;
		inorder_travel(root, res);
		return res;
	}

	void inorder_travel(TreeNode* root, vector<int>& res) {
		if (!root) return;
		
		inorder_travel(root->left, res);
		res.emplace_back(root->val);
		inorder_travel(root->right, res);
		return;
	}
	/*=====================================================================================*/

	/*================     Binary Tree postorderTraversal  220630 11：13    ==================*/
	vector<int> postorderTraversal(TreeNode* root) {
		vector<int> res;
		postorder_travel(root, res);
		return res;
	}

	void postorder_travel(TreeNode* root, vector<int>& res) {
		if (!root) return;

		postorder_travel(root->left, res);
		postorder_travel(root->right, res);
		res.emplace_back(root->val);
		return;
	}
	/*=====================================================================================*/

	/*=============     Binary Tree Level Order Traversal  220701 16:16    ==================*/
	vector<vector<int>> levelOrder(TreeNode* root) {
		if (!root) return vector<vector<int>>{};
		vector<vector<int>> result;
		
		TreeNode* node = root;
		queue<TreeNode*> q;
		q.emplace(node);

		while (!q.empty()) {

			vector<TreeNode*> level_vec;
			vector<int> level_val;

			while (!q.empty()) {
				TreeNode* node_temp = q.front();
				q.pop();
				level_val.emplace_back(node_temp->val);
				level_vec.emplace_back(node_temp);
			}

			result.emplace_back(level_val);

			for(auto au: level_vec) {
				if(au->left)
					q.emplace(au->left);
				if (au->right)
					q.emplace(au->right);
			}
		}
		

		return result;
	}

	void level_travel(TreeNode* node, int level, vector<vector<int>>& result) {
		if (!node) return;

		if (result.size() < level + 1) {
			result.resize(level + 1);
		}

		result[level].emplace_back(node->val);

		level_travel(node->left, level + 1, result);
		level_travel(node->right, level + 1, result);
	}
	/*========================================================================================*/

	/*===========================  Symmetric Tree  220704 08:58 ==============================*/
	bool isSymmetric(TreeNode* root) {
		if (!root->left && !root->right) return true;
		if (!root->left || !root->right) return false;

		bool rest = isSymmetric_sub(root->left, root->right);

		return rest;
	}

	bool isSymmetric_sub(TreeNode* node_left, TreeNode* node_right) {
		if (!node_left && !node_right) {
			return true;
		}

		if (!node_left && node_right) {
			return false;
		}

		if (node_left && !node_right) {
			return false;
		}

		if (node_left->val != node_right->val) return false;

		return isSymmetric_sub(node_left->left, node_right->right) && isSymmetric_sub(node_left->right, node_right->left);

	}
	/*========================================================================================*/

	/*=========================     Path Sum  220704 10:06     ===============================*/
	bool hasPathSum(TreeNode* root, int targetSum) {
		if (!root) return false;

		bool rest = hasPathSum_sub(root, targetSum, 0);

		return rest;
	}

	bool hasPathSum_sub(TreeNode* node, int& targetSum, int sum) {
		if (!node) {
			return false;
		}

		if (!node->left && !node->right && sum + node->val == targetSum) {
			return true;
		}

		return hasPathSum_sub(node->left, targetSum, sum + node->val) || hasPathSum_sub(node->right, targetSum, sum + node->val);
	}

	/*========================================================================================*/

	/*==== Construct Binary Tree from Inorder and Postorder Traversal 220705 10:56 ===========*/
	TreeNode* buildTree_inorder_postorder(vector<int>& inorder, vector<int>& postorder) {

		TreeNode* root = build_btree_inorder_postorder(inorder, postorder);

		return root;
	}

	TreeNode* build_btree_inorder_postorder(vector<int> inorder, vector<int> postorder) {

		TreeNode* node = nullptr;

		if (inorder.empty()) {
			return node;
		}

		int n = inorder.size();

		node = new TreeNode(postorder.back());

		if (n == 1) {
			return node;
		}

		int od_left = find(inorder.begin(), inorder.end(), postorder.back()) - inorder.begin();

		//if (postorder.back() == inorder.back()) {
		//	vector<int> inorder_left(inorder.begin(), inorder.begin() + od_left);
		//	vector<int> postorder_left(postorder.begin(), postorder.begin() + od_left);
		//	node->left = build_btree_inorder_postorder(inorder_left, postorder_left);
		//	return node;
		//}
		//else if (postorder.back() == inorder.front()) {
		//	vector<int> inorder_right(inorder.begin() + od_left + 1, inorder.end());
		//	vector<int> postorder_right(postorder.begin() + od_left, postorder.end() - 1);
		//	node->right = build_btree_inorder_postorder(inorder_right, postorder_right);
		//	return node;
		//}
		//else {
		//	//vector<int>::iterator od_postorder = find(postorder.begin(), postorder.end(), inorder[od_inorder - inorder.begin() + 1]);
		//	
		//	vector<int> inorder_left(inorder.begin(), inorder.begin() + od_left);
		//	vector<int> inorder_right(inorder.begin() + od_left + 1, inorder.end());

		//	vector<int> postorder_left(postorder.begin(), postorder.begin() + od_left);
		//	vector<int> postorder_right(postorder.begin() + od_left, postorder.end() - 1);

		//	node->left = build_btree_inorder_postorder(inorder_left, postorder_left);
		//	node->right = build_btree_inorder_postorder(inorder_right, postorder_right);

		//	return node;
		//}

		if (postorder.back() != inorder.front()) {
			vector<int> inorder_left(inorder.begin(), inorder.begin() + od_left);
			vector<int> postorder_left(postorder.begin(), postorder.begin() + od_left);
			node->left = build_btree_inorder_postorder(inorder_left, postorder_left);
		}

		if (postorder.back() != inorder.back()) {
			vector<int> inorder_right(inorder.begin() + od_left + 1, inorder.end());
			vector<int> postorder_right(postorder.begin() + od_left, postorder.end() - 1);
			node->right = build_btree_inorder_postorder(inorder_right, postorder_right);
		}

		return node;
	}
	/*========================================================================================*/

	/*==== Construct Binary Tree from Preorder and Inorder Traversal 220706 10:39 ===========*/
	TreeNode* buildTree_preorder_inorder(vector<int>& preorder, vector<int>& inorder) {

		TreeNode* root = build_btree_preorder_inorder(preorder, inorder);

		return root;
	}

	TreeNode* build_btree_preorder_inorder(vector<int> preorder, vector<int> inorder) {

		if (inorder.empty()) return nullptr;

		TreeNode* node_root = new TreeNode(preorder.front());//root

		int od_left = find(inorder.begin(), inorder.end(), preorder.front()) - inorder.begin();

		if (preorder.front() != inorder.front()) {
			vector<int> preorder_left(preorder.begin() + 1, preorder.begin() + od_left + 1);
			vector<int> inorder_left(inorder.begin(), inorder.begin() + od_left);
			
			node_root->left = build_btree_preorder_inorder(preorder_left, inorder_left);
		}

		if (preorder.front() != inorder.back()) {
			vector<int> preorder_right(preorder.begin() + 1 + od_left, preorder.end());
			vector<int> inorder_right(inorder.begin() + od_left + 1, inorder.end());
			
			node_root->right = build_btree_preorder_inorder(preorder_right, inorder_right);
		}

		return node_root;
	}
	/*========================================================================================*/

	/*=========     Populating Next Right Pointers in Each Node 220707 13:23     =============*/
	TreeNode* connect_next_right(TreeNode* root) {
		if (!root) return nullptr;
		queue<TreeNode*> q;
		q.emplace(root);

		while (!q.empty()) {
			TreeNode* node_temp = q.front();
			q.pop();
			int n = q.size();

			node_temp->next = nullptr;
			if (node_temp->left) {
				q.emplace(node_temp->left);
			}
			if (node_temp->right) {
				q.emplace(node_temp->right);
			}

			while (n > 0) {
				n--;
				q.front()->next = nullptr;
				node_temp->next = q.front();
				q.pop();
				node_temp = node_temp->next;
				if (node_temp->left) {
					q.emplace(node_temp->left);
				}
				if (node_temp->right) {
					q.emplace(node_temp->right);
				}
			}
		}

		return root;
	}
	/*========================================================================================*/

	/*===========     Lowest Common Ancestor of a Binary Tree 220711 10:36     ===============*/
	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
		if (!root || root == p || root == q) {
			return root;
		}

		TreeNode* left_sub = lowestCommonAncestor(root->left, p, q);
		TreeNode* right_sub = lowestCommonAncestor(root->right, p, q);

		if (left_sub && right_sub) {
			return root;
		}
		else if (left_sub && !right_sub) {
			return left_sub;
		}
		else if (!left_sub && right_sub) {
			return right_sub;
		}

		return nullptr;
	}
	/*========================================================================================*/

	/*============     Serialize and Deserialize Binary Tree 220712 10:51     ================*/
	// Encodes a tree to a single string.
	string serialize(TreeNode* root) {
		string data;
		/*if (!root) {
			data += "null";
			return data;
		}

		queue<TreeNode*> q;
		
		q.emplace(root);

		while (!q.empty()) {
			int n = q.size();

			while (n > 0) {
				n--;
				TreeNode* node_temp = q.front();
				q.pop();

				if (node_temp) {
					data += to_string(node_temp->val) + ",";
				}
				else {
					data += "null,";
					continue;
				}

				if (node_temp->left) 
					q.emplace(node_temp->left);
				else 
					q.emplace(nullptr);

				if (node_temp->right) 
					q.emplace(node_temp->right);
				else 
					q.emplace(nullptr);
			}
		}

		for (int i = data.size() - 1; i >= 0; i--) {
			if (data[i] == ',' || data[i] == 'n' || data[i] == 'u' || data[i] == 'l') data.pop_back();
			else break;
		}

		return data;*/

		data = serialize_0(root);

		return data;
	}

	string serialize_0(TreeNode* root) {
		if (!root) return "#";

		return to_string(root->val)+","+ serialize_0(root->left) +"," + serialize_0(root->right);
	}

	// Decodes your encoded data to tree.
	TreeNode* deserialize(string data) {
		//TreeNode* root = nullptr;

		if (data == "null") return nullptr;
		

		vector<string> data_vec;
		queue<TreeNode*> q;
		int pos = 0;
		while (( pos = data.find(",")) != std::string::npos) {
			data_vec.emplace_back(data.substr(0,pos));
			data.erase(0, 1 + data_vec.back().size());
		}
		data_vec.emplace_back(data);

		TreeNode* root = new TreeNode(stoi(data_vec.front()));
		q.emplace(root);

		int n = data_vec.size();
		int i = 0;

		while(i < n && !q.empty()) {

			Solution::TreeNode* temp_node = q.front();
			q.pop();

			i++;
			if (i < n && data_vec[i] != "null") {
				temp_node->left = new TreeNode(stoi(data_vec[i]));
				q.emplace(temp_node->left);
			}

			i++;
			if (i < n && data_vec[i] != "null") {
				temp_node->right = new TreeNode(stoi(data_vec[i]));
				q.emplace(temp_node->right);
			}
		}

		return root;
	}
	/*========================================================================================*/

	/*============     Number of Islands 220714 11:57   220720 11:05   ================*/
	void DFS_islands(int i, int j, vector<vector<char>>& grid) {
		int m = grid.size();
		int n = grid[0].size();

		if (i<0 || i>=m || j<0 || j>=n || grid[i][j] == '0') return;
		
		grid[i][j] = '0';
		DFS_islands(i-1, j, grid);
		DFS_islands(i+1, j, grid);
		DFS_islands(i, j-1, grid);
		DFS_islands(i, j+1, grid);
	}

	int numIslands(vector<vector<char>>& grid) {
		int m = grid.size();
		int n = grid[0].size();
		int num = 0;

		/*===============  DFS  =====================*/
		for (int i = 0; i < m; i++) 
			for (int j = 0; j < n; j++) 
				if (grid[i][j] == '1') {
					num++;
					DFS_islands(i, j, grid);
				}
		return num;

		/*===============  BFS  =====================*/
		/*queue<pair<int, int>> q;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (grid[i][j] == '0') continue;
				grid[i][j] = '0';
				q.emplace(i, j);
				num++;
				while (!q.empty()) {
					int x = q.front().first;
					int y = q.front().second;
					
					q.pop();
					if (x > 0 && grid[x-1][y] == '1') {
						grid[x - 1][y] = '0';
						q.emplace(x-1,y);
					}
					if (x < m - 1 && grid[x + 1][y] == '1') {
						grid[x + 1][y] = '0';
						q.emplace(x + 1, y);
					}
					if (y > 0 && grid[x][y-1] == '1') {
						grid[x][y-1] = '0';
						q.emplace(x, y-1);
					}
					if (y < n - 1 && grid[x][y + 1] == '1') {
						grid[x][y+1] = '0';
						q.emplace(x, y + 1);
					}
				}
			}
		}
		return num;*/
	}
	/*========================================================================================*/

    /*=======================         Open the Lock 220716 19:07       =======================*/
	int openLock(vector<string>& deadends, string target) {

		if (target == "0000") return 0;
		
		queue<string> q;
		map<string, int> map_dead;
		for (auto au : deadends)
			map_dead[au]++;

		if (map_dead["0000"]) return -1;

		int step_min = 0;
		q.emplace("0000");
		
		while (!q.empty()) {
			step_min++;
			int n = q.size();
			for (int i = 0; i < n; i++) {
				string str_temp = q.front();
				q.pop();
			
				for (int j = 0; j < 4; j++) {
					string str = str_temp;
					str[j] = (str_temp[j] - '0' - 1 + 10) % 10 + '0';
					if (!map_dead[str]) {
						map_dead[str]++;
						if (str == target) 
							return step_min;
						q.emplace(str);
					}

					str[j] = (str_temp[j] - '0' + 1) % 10 + '0';
					if (!map_dead[str]) {
						map_dead[str]++;
						if (str == target) 
							return step_min;
						q.emplace(str);
					}
				}
			}
		}
		return -1;
	}
	/*========================================================================================*/

	/*=======================         Perfect Squares 220717 19:15       =======================*/
	int numSquares(int n) {
		//if(is_square(n)) return 1;
	  //while ((n % 4) == 0){ // n%4 == 0  
		  //n /= 4;
	  //}
	  //if ((n % 8) == 7){ // n%8 == 7
		  //return 4;
	  //}
	  //int sq = sqrt(n);
	  //for(int i = 1; i <= sq; i++){
		  //int l = n - i * i;
		  //if(is_square(l)){
			  //return 2;
		  //}
	  //}

	  //return 3;
	 // ///////////////////////////////////////////////////////////////////
		//if (n <= 0) {
		//	return 0;
		//}

		//vector<int> result(n + 1, INT_MAX);
		//result[0] = 0;

		//for (int i = 1; i <= n; i++) {
		//	for (int j = 1; j * j <= i; j++) {
		//		int dow = j * j;
		//		result[i] = min(result[i], result[i - dow] + 1);
		//	}
		//}

		//return result.back();
		///////////////////////////////////////////////////////////////
		int count = 0;
		vector<int> square_vector;
		queue<int> square_queue;
		for (int i = 1; i*i <= n;i++) 
			square_vector.emplace_back(i * i);
		std::reverse(square_vector.begin(), square_vector.end());

		square_queue.emplace(n);
		while (!square_queue.empty()) {
			count++;
			int m = square_queue.size();
			while (m>0) {
				m--;
				int num = square_queue.front();
				square_queue.pop();
				for (auto au : square_vector) {
					if (num < au) 
						continue;
					if (num - au == 0) 
						return count;
					square_queue.emplace(num - au);
				}
			}
		}

		return n;
	}
	/*========================================================================================*/

	/*=======================         Target Sum 220722 08:41       =======================*/
	int findTargetSumWays(vector<int>& nums, int target) {
		int current = 0;
		int i = 0;
		int result = find_target_sum(nums, target, i, current);
		return result;
	}

	int find_target_sum(vector<int>& nums, int& target, int i, int current) {

		if (i == nums.size()) {
			if (current != target) return 0;
			return 1;
		}
		return find_target_sum(nums, target, i + 1, current + nums[i]) + find_target_sum(nums, target, i + 1, current - nums[i]);
		
	}
	/*========================================================================================*/

	/*=======================         Flood Fill 220728 09:04         =======================*/
	vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
		int original_color = image[sr][sc];
		recursive_floodFill(image, original_color, sr, sc, color);

		return image;
	}
	
	void recursive_floodFill(vector<vector<int>>& image, int& original_color, int sr, int sc, int& color) {
		int m = image.size();
		int n = image[0].size();

		if (sr < 0 || sc < 0 || sr >= m || sc >= n || image[sr][sc] != original_color || image[sr][sc] == color) {
			return;
		}

		if (image[sr][sc] == original_color) {

			image[sr][sc] = color;
			recursive_floodFill(image, original_color, sr - 1, sc, color);
			recursive_floodFill(image, original_color, sr + 1, sc, color);
			recursive_floodFill(image, original_color, sr, sc-1, color);
			recursive_floodFill(image, original_color, sr, sc+1, color);
		}

		return;
	}
	/*========================================================================================*/

	/*=======================         01 Matrix 220729 10:31         =======================*/

	vector<vector<int>> updateMatrix_0(vector<vector<int>>& mat) {
		int m = mat.size();
		int n = mat[0].size();
		vector<vector<int>> res(m, vector<int>(n, INT_MAX-100));

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (!mat[i][j]) {
					res[i][j] = 0;
					continue;
				}
				if (i-1>=0) {
					res[i][j] = min(res[i][j], res[i - 1][j]+1);
				}
				if (j - 1 >= 0) {
					res[i][j] = min(res[i][j], res[i][j - 1]+1);
				}
				
			}
		}

		for (int i = m - 1; i >= 0; i--) {
			for (int j = n - 1; j >= 0; j--) {
				if (!mat[i][j]) {
					res[i][j] = 0;
					continue;
				}
				if (i + 1 < m) {
					res[i][j] = min(res[i][j], res[i + 1][j]+1);
				}
				if (j + 1 < n) {
					res[i][j] = min(res[i][j], res[i][j + 1]+1);
				}

			}
		}

		return res;
	}

	vector<vector<int>> updateMatrix(vector<vector<int>>& mat) {
		int m = mat.size();
		int n = mat[0].size();

		vector<vector<int>> res(m,vector<int>(n, INT_MAX));
		queue<pair<int, int>> q;

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (mat[i][j] == 0) {
					q.emplace(i,j);
					res[i][j] = 0;
				}
			}
		}

		while (!q.empty()) {
			int x = q.front().first;
			int y = q.front().second;
			q.pop();

			if (x - 1 >= 0 && res[x - 1][y] > res[x][y] + 1) {
				res[x - 1][y] = res[x][y] + 1;
				q.emplace(x - 1, y);
			}
			if (x + 1 < m && res[x + 1][y] > res[x][y] + 1) {
				res[x + 1][y] = res[x][y] + 1;
				q.emplace(x + 1, y);
			}
			if (y - 1 >= 0 && res[x][y - 1] > res[x][y] + 1) {
				res[x][y-1] = res[x][y] + 1;
				q.emplace(x, y - 1);
			}
			if (y + 1 < n && res[x][y + 1] > res[x][y] + 1) {
				res[x][y + 1] = res[x][y] + 1;
				q.emplace(x, y + 1);

			}
		}

		return res;
	}
	/*========================================================================================*/

	/*=======================         Keys and Rooms 220730 18:45         =======================*/
	bool canVisitAllRooms_0(vector<vector<int>>& rooms) {
		int n = rooms.size();
		int m = rooms[0].size();
		bool res = false;
		map<int, int> visit;
		stack<int> st;

		visit[0]++;
		st.emplace(0);
		
		while (!st.empty()) {
			int next = st.top();
			st.pop();
			for (auto au : rooms[next]) {
				if (!visit[au]) {
					visit[au]++;
					st.emplace(au);
				}
			}
		}

		for (int i = 0; i < n; i++) {
			if (!visit[i])
				return false;
		}

		return true;
	}


	bool canVisitAllRooms(vector<vector<int>>& rooms) {
		int n = rooms.size();
		int m = rooms[0].size();
		map<int, int> visit;
		
		
		visit[0]++;
		for (int i = 0; i < m; i++) {
			if (recursive_canVisitAllRooms(rooms, visit, rooms[0][i])) 
				return true;
		}

		for (int i = 0; i < n; i++)
			if (!visit[i]) 
				return false;
		
		return true;
	}

	bool recursive_canVisitAllRooms(vector<vector<int>>& rooms, map<int, int>& visit, int i) {
		visit[i]++;

		if (visit.size() == rooms.size()) return true;

		for (int j = 0; j < rooms[i].size(); j++) {
			if (!visit[rooms[i][j]] && recursive_canVisitAllRooms(rooms, visit, rooms[i][j])) return true;
		}

		return false;
	}
	/*========================================================================================*/

	/*=======================         Keys and Rooms 220801 09:55         =======================*/
	TreeNode* searchBST(TreeNode* root, int val) {
		return recursive_searchBST(root, val);
	}

	TreeNode* recursive_searchBST(TreeNode* root, int& val) {

		if (!root)
			return nullptr;
		else if (root->val == val)
			return root;

		TreeNode* res = nullptr;

		if (root->left)
			res = recursive_searchBST(root->left, val);
		
		if (!res && root->right) 
			return recursive_searchBST(root->right, val);
		
		return res;
	}
	/*========================================================================================*/

	/*=======================         Pascal's Triangle II 220802 12:13         =======================*/
	vector<int> getRow(int rowIndex) {
		vector<int> row_vector = {1};

		row_vector = get_pascal_triangle(rowIndex, row_vector);

		return row_vector;
	}

	vector<int> get_pascal_triangle(int& rowIndex, vector<int> row_vector) {

		int n = row_vector.size();

		if (rowIndex + 1 == n) 
			return row_vector;
		

		vector<int> temp_vector(1, 1);
		for (int i = 0; i < n - 1; i++)
			temp_vector.emplace_back(row_vector[i] + row_vector[i + 1]);
		
		temp_vector.emplace_back(1);

		return get_pascal_triangle(rowIndex, temp_vector);
	}
	/*========================================================================================*/

	/*=======================        Climbing Stairs 220803 13:23       =======================*/
	int climbStairs(int n) {
		vector<int> v(n + 1, 1);
		v[2] = 2;
		for (int i = 2; i <= n; i++) {
			v[i] = v[i - 1] + v[i - 2];
		}
		return v[n];
	}
	/*========================================================================================*/

	/*===================        K-th Symbol in Grammar 220804 13:50       ===================*/
	int kthGrammar(int n, int k) {
		if (n == 1) return 0;
		vector<int> v{ 0,1 };
		return recursive_kthGrammar(n, k, v);
	}

	int recursive_kthGrammar(int n, int k, vector<int>& v) {
		n--;

		if (n == 1)
			return v[k - 1];

		int l = pow(2, n-1);
		if (k > l) {
			k -= l;
			if (v[1] == 1) 
				v = { 1,0 };
			else 
				v = { 0,1 };
		}
		else
			if (v[0] == 1) 
				v = { 1,0 };
			else 
				v = { 0,1 };
		
		return recursive_kthGrammar(n, k, v);
	}
	/*========================================================================================*/

	/*===============     Unique Binary Search Trees II 220805 12:35       ===================*/
	vector<TreeNode*> generateTrees(int n) {
		return recursive_generateTrees_0(1, n);
	}

	vector<TreeNode*> recursive_generateTrees_0(int start, int end) {
		vector<TreeNode*> ans;

		if (start > end) {
			ans.push_back(nullptr);
			return ans;
		}

		for (int i = start; i <= end; i++) {
			vector <TreeNode*> left = recursive_generateTrees_0(start, i - 1);
			vector <TreeNode*> right = recursive_generateTrees_0(i + 1, end);

			for (auto l : left)
				for (auto r : right) {
					TreeNode* newnode = new TreeNode(i, l, r);
					ans.push_back(newnode);
				}
		}

		return ans;
	}

	void recursive_generateTrees(int i, int n, int limit, vector<int> l_vector, vector<int> r_vector, vector<int> res_vector, vector<vector<int>>& res_all) {
		
		if (i < 1 || i > limit) {
			res_all.emplace_back(res_vector);
			return;
		}

		for (int j = 0; j < l_vector.size(); j++) {
			res_vector.emplace_back(l_vector[j]);
			for (int k = 0; k < r_vector.size(); k++) {
				
			}
		}

		return;
	}
	/*========================================================================================*/

	/*======================     Sort an Array 220805 18:48       ============================*/
	vector<int> sortArray(vector<int>& nums) {
		int n = nums.size() / 2;
		if (!n) return nums;

		vector<int> res = recursive_sortArray(vector<int>(nums.begin(), nums.begin() + n), vector<int>(nums.begin() + n, nums.end()));
		return res;
	}

	vector<int> recursive_sortArray(vector<int> left, vector<int> right) {
		
		if (left.size() > 1) {
			int n_l = left.size() / 2;
			left = recursive_sortArray(vector<int>(left.begin(), left.begin() + n_l), vector<int>(left.begin() + n_l, left.end()));
		}

		if (right.size() > 1) {
			int n_r = right.size() / 2;
			right = recursive_sortArray(vector<int>(right.begin(), right.begin() + n_r), vector<int>(right.begin() + n_r, right.end()));
		}

		return merge_sortArray(left, right);
	}

	vector<int> merge_sortArray(vector<int> left, vector<int> right) {
		int i = 0;
		int j = 0;
		int n = left.size();
		int m = right.size();
		vector<int> res;

		while (i < n && j < m) {
			if (i < n && left[i] < right[j]) 
				res.emplace_back(left[i++]);
			else if(j < m) 
				res.emplace_back(right[j++]);
		}

		while (i<n) 
			res.emplace_back(left[i++]);
		
		while (j < m) 
			res.emplace_back(right[j++]);
		
		return res;
	}
	/*========================================================================================*/

	/*======================      N-Queens II 220807 15:14        ============================*/
	int totalNQueens(int n) {
		int count = 0;
		vector<vector<int>> board(n, vector<int>(n, 0));
		totalNQueens(0, board, count);
		return count;
	}

	void totalNQueens(int i, vector<vector<int>>& board, int& count) {
		int n = board.size();

		for (int j = 0; j < n; j++) {
			if (is_not_attack(i, j, board)) {
				put_chess(i, j, board);

				if (i == n - 1) 
					count++;
				else 
					totalNQueens(i + 1, board, count);

				remove_chess(i, j, board);
			}
		}
	}

	bool is_not_attack(int& i,int& j, vector<vector<int>>&  board) {
		return !board[i][j];
	}

	void put_chess(int i, int j, vector<vector<int>>& board) {
		int n = board.size();

		board[i][j] = i * n + j + 1;

		for (int k = 0; k < n; k++)
			if (!board[k][j]) board[k][j] = board[i][j];

		for (int k = 0; k < n; k++)
			if (!board[i][k]) board[i][k] = board[i][j];

		int k = i;
		int l = j;

		while (++k < n && ++l < n)
			if (!board[k][l]) board[k][l] = board[i][j];

		k = i;
		l = j;

		while (--k >= 0 && --l >= 0)
			if (!board[k][l]) board[k][l] = board[i][j];

		k = i;
		l = j;

		while (--k >= 0 && ++l < n)
			if (!board[k][l]) board[k][l] = board[i][j];

		k = i;
		l = j;

		while (++k < n  && --l >= 0)
			if (!board[k][l]) board[k][l] = board[i][j];
	}

	void remove_chess(int i, int j, vector<vector<int>>& board) {
		int n = board.size();

		board[i][j] = 0;

		for (int k = 0; k < n; k++)
			if (board[k][j] == i * n + j + 1) board[k][j] = 0;

		for (int k = 0; k < n; k++)
			if (board[i][k] == i * n + j + 1) board[i][k] = 0;

		int k = i;
		int l = j;

		while (++k < n && ++l < n)
			if (board[k][l] == i * n + j + 1) board[k][l] = 0;

		k = i;
		l = j;

		while (--k >= 0 && --l >= 0)
			if (board[k][l] == i * n + j + 1) board[k][l] = 0;

		k = i;
		l = j;

		while (--k >= 0 && ++l < n)
			if (board[k][l] == i * n + j + 1) board[k][l] = 0;

		k = i;
		l = j;

		while (++k < n  && --l >= 0)
			if (board[k][l] == i * n + j + 1) board[k][l] = 0;
	}
	/*========================================================================================*/

	/*======================      Sudoku Solver 220808 12:22      ============================*/
	void solveSudoku(vector<vector<char>>& board) {
		vector<int> row_begin;
		vector<int> col_begin;
		bool is_final = false;

		for (int i = 0; i <= 9; i += 3) {
			row_begin.emplace_back(i);
			col_begin.emplace_back(i);
		}
		
		recursive_solveSudoku(board, row_begin, col_begin, 0, 0, is_final);
	}

	void recursive_solveSudoku(vector<vector<char>>& board, vector<int>& row_begin, vector<int>& col_begin, int row, int col, bool& is_final) {
		vector<char> candidate{ '1', '2', '3', '4', '5', '6', '7', '8', '9' };

		if (row == 9) {
			is_final = true;
			return;
		}
		else if (col == 9 && !is_final) 
			recursive_solveSudoku(board, row_begin, col_begin, row + 1, 0, is_final);
		else if (row < 9 && col < 9 && board[row][col] != '.' && !is_final) 
			recursive_solveSudoku(board, row_begin, col_begin, row, col + 1, is_final);
		else if (!is_final) 
			for (auto au: candidate) 
				if (is_valid(board, row_begin, col_begin, au, row, col)) {
					board[row][col] = au;
					recursive_solveSudoku(board, row_begin, col_begin, row, col + 1, is_final);
					if (is_final) return;
					board[row][col] = '.';
				}
	}

	bool is_valid(vector<vector<char>>& board, vector<int>& row_begin, vector<int>& col_begin, char candidate_char, int row, int col) {

		for (int i = 0; i < 9; i++)
			if (i != row && board[i][col] == candidate_char)
				return false;
			
		for (int j = 0; j < 9; j++)
			if (j != col && board[row][j] == candidate_char)
				return false;

		int begin_i;
		int begin_j;

		for (int i = 0; i < 3; i++)
			if (row >= row_begin[i] && row < row_begin[i + 1]) 
				begin_i = row_begin[i];
			
		for (int j = 0; j < 3; j++)
			if (col >= col_begin[j] && col < col_begin[j + 1]) 
				begin_j = col_begin[j];
		
		for (int i = begin_i; i < begin_i + 3; i++)
			for (int j = begin_j; j < begin_j + 3; j++) 
				if (row != i && col != j && board[i][j] == candidate_char) 
					return false;
				
		return true;
	}
	/*========================================================================================*/

	/*======================      Combinations 220809 10:35      ============================*/
	vector<vector<int>> combine(int n, int k) {

		vector<int> vec; 
		vector<vector<int>> res;
		bool is_finish = false;

		backtrack_combine(n, k, 1, 1, vec, res, is_finish);

		return res;
	}

	void backtrack_combine(int& n, int& k, int i, int j, vector<int> vec, vector<vector<int>>& res, bool& is_finish) {

		if (i + k - 1 > n) {
			is_finish = true;
			return;
		}

		if (vec.size() == k) {
			res.emplace_back(vec);
			return;
		}

		for (int l = j; l <= n && !is_finish; l++) {
			vec.emplace_back(l);

			backtrack_combine(n, k, i, l + 1, vec, res, is_finish);

			if (is_finish) return;

			vec.pop_back();
		}

		if (vec.size() == 1 && !is_finish) {
			vec.clear();
			backtrack_combine(n, k, i + 1, i + 1, vec, res, is_finish);
		}
			
	}

	/*========================================================================================*/

	/*======================      generateParenthesis 220810 11:35      ============================*/
	vector<string> generateParenthesis(int& n) {
		int left = 0;
		int right = 0;
		string res;
		vector<string> res_v;

		backtrack_generateParenthesis(n, 0, 0, res, res_v);

		return res_v;
	}

	void backtrack_generateParenthesis(int& n, int left, int right, string& res, vector<string>& res_v) {
		if (res.size() == n << 1) {
			res_v.emplace_back(res);
			return;
		}

		if (left < n) {
			res += '(';
			backtrack_generateParenthesis(n, left + 1, right, res, res_v);
			res.pop_back();
		}

		if (left > right) {
			res += ')';
			backtrack_generateParenthesis(n, left, right+1, res, res_v);
			res.pop_back();
		}
	}
	/*========================================================================================*/

	/*==============      Largest Rectangle in Histogram 220811 10:31      ===================*/
	int myAtoi(string s) {
		int area_max = INT_MIN;
		int left = 0;
		int right = 0;
		int height_min = INT_MAX;
		recursive_myAtoi(s, 0, 0, height_min, area_max);
		
		return area_max;
	}

	void recursive_myAtoi(string& s, int left, int right, int height_min, int &area_max) {
		int n = s.size();

		if (left == n) {
			area_max = max(area_max, s[left] - '0');
			return;
		}
		else if (right == n) {
			height_min = INT_MAX;
			recursive_myAtoi(s, left + 1, left + 1, height_min, area_max);
		}
		else {
			height_min = min(height_min, s[right] - '0');
			area_max = max(area_max, (right - left + 1) * height_min);
			recursive_myAtoi(s, left, right + 1, height_min, area_max);
		}
	}

	int largestRectangleArea(vector<int>& heights) {
		//int area_max = INT_MIN;
		//int left = 0;
		//int right = 0;
		//int height_min = INT_MAX;
		////recursive_largestRectangleArea(heights, 0, 0, height_min, area_max);
		//int n = heights.size();
		//if (n == 1) return heights.front();
		//int res;
		//res = fast_largestRectangleArea(heights, 0, n-1, height_min, area_max);
		//res = max(res, area_max);

		int n = heights.size();
		int area_max = 0;
		stack<int> st;

		for (int i = 0; i <= n; i++) {

			int height_i = 0;//i==n, height is 0
			if (i < n) height_i = heights[i];

			while (!st.empty() && height_i < heights[st.top()]) {

				int od_top = st.top();

				st.pop();

				int width = i;
				if (!st.empty()) {
					width = i - st.top() - 1;
				}

				int area = heights[od_top] * width;
				area_max = max(area_max, area);
			}

			st.emplace(i);
		}


		return area_max;
	}

	int fast_largestRectangleArea(vector<int>& heights, int left, int right, int height_min, int& area_max) {
		int n = heights.size();

		if (left >= right) {
			if(left >= n || left< 0)
				return max(heights[right], area_max);
			else if(right >= n || right < 0)
				return max(heights[left], area_max);
			return max(heights[right], area_max);
		}

		height_min = heights[left];
		int od_min = left;
		int count = 0;
		for (int i = left; i <= right; i++) {
			if (height_min > heights[i]) {
				od_min = i;
				height_min = heights[i];
			}

			if (height_min == heights[i]) {
				count++;
			}
		}

		if (heights[left] == heights[right] && left == od_min && count == right - left + 1) {
			return area_max = max(area_max, (right - left + 1) * heights[od_min]);
		}

		area_max = max(area_max, (right - left + 1) * heights[od_min]);
		
		count = od_min;

		while (++count < n && heights[od_min] == heights[count]) {
		}

		return max(fast_largestRectangleArea(heights, left, od_min - 1, height_min, area_max), fast_largestRectangleArea(heights, count, right, height_min, area_max));
	}

	void recursive_largestRectangleArea(vector<int>& heights, int left, int right, int height_min, int& area_max) {
		int n = heights.size();

		if (left == n) {
			//area_max = max(area_max, heights[left]);
			return;
		}
		else if (right == n) {
			height_min = INT_MAX;
			recursive_largestRectangleArea(heights, left + 1, left + 1, height_min, area_max);
		}
		else {
			height_min = min(height_min, heights[right]);
			area_max = max(area_max, (right - left + 1) * height_min);
			recursive_largestRectangleArea(heights, left, right + 1, height_min, area_max);
		}
	}
	/*========================================================================================*/

	/*========================     Permutations 220812 15:23      ============================*/
	vector<vector<int>> permute(vector<int>& nums) {
		vector<int> res;
		vector<vector<int>> res_v;
		backtrack_permute(nums, 0, res, res_v);

		return res_v;
	}

	void backtrack_permute(vector<int>& nums, int begin, vector<int> res, vector<vector<int>>& res_v) {
		int n = nums.size();

		if (begin == n) {
			res_v.emplace_back(nums);
			return;
		}

		for (int i = begin; i < n; i++) {
			swap(nums[begin], nums[i]);
			backtrack_permute(nums, begin+1, res, res_v);
			swap(nums[i], nums[begin]);
		}

		return;
	}
	/*========================================================================================*/

	/*===========     Letter Combinations of a Phone Number 220814 19:25      ================*/
	vector<string> letterCombinations(string digits) {
		if (digits.empty()) return vector<string>{};
		vector<string> res_v;
		vector<string> letter = {"#","#", "abc","def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
		vector<int> od;
		string res;

		for (auto au : digits) {
			od.emplace_back(au - '0');
		}

		backtrack_letterCombinations(od, letter, 0, res, res_v);

		return res_v;
	}

	void backtrack_letterCombinations(vector<int>& od, vector<string>& letter, int id, string res, vector<string>& res_v) {
		int n = od.size();
		if (res.size() == n) {
			res_v.emplace_back(res);
			return;
		}

		for (auto au : letter[od[id]]) {
			res.push_back(au);
			backtrack_letterCombinations(od, letter, id + 1, res, res_v);
			res.pop_back();
		}
	}
	/*========================================================================================*/

	/*======================     The Skyline Problem 220815 10:28      =======================*/
	vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
		vector<vector<int>> res_v;

		res_v = dc_rectangle(buildings, 0, buildings.size() - 1);

		return res_v;
	}

	vector<vector<int>> dc_rectangle(vector<vector<int>>& trans_v, int begin, int end) {

		if (begin == end) 
			return { {trans_v[begin][0], trans_v[begin][2]}, {trans_v[begin][1], 0}};

		int middle_od = (begin + end) >> 1;

		vector<vector<int>> trans_v_left = dc_rectangle(trans_v, begin, middle_od);
		vector<vector<int>> trans_v_right = dc_rectangle(trans_v, middle_od + 1, end);

		return merge_rectangle(trans_v_left, trans_v_right);
	}

	vector<vector<int>> merge_rectangle(vector<vector<int>> trans_v_left, vector<vector<int>> trans_v_right) {
		vector<vector<int>> merge_v;
		int n = trans_v_left.size();
		int m = trans_v_right.size();

		int leftPos = 0;
		int rightPos = 0;

		int leftPrevHeight = 0;
		int rightPrevHeight = 0;

		int curX = 0;
		int curY = 0;

		while (leftPos < n && rightPos < m) {
			int nextLeftX = trans_v_left[leftPos][0];
			int nextRightX = trans_v_right[rightPos][0];
			if (nextLeftX < nextRightX) {
				curX = nextLeftX;
				curY = max(trans_v_left[leftPos][1], rightPrevHeight);
				leftPrevHeight = trans_v_left[leftPos++][1];
			}
			else if (nextLeftX > nextRightX) {
				curX = nextRightX;
				curY = max(trans_v_right[rightPos][1], leftPrevHeight);
				rightPrevHeight = trans_v_right[rightPos++][1];
			}
			else {
				curX = nextLeftX;
				curY = max(trans_v_left[leftPos][1], trans_v_right[rightPos][1]);
				leftPrevHeight = trans_v_left[leftPos++][1];
				rightPrevHeight = trans_v_right[rightPos++][1];
			}

			if (merge_v.empty() || curY != merge_v.back()[1]) {
				merge_v.emplace_back(vector<int>{ curX, curY });
			}
		}

		while (leftPos < trans_v_left.size()) {
			merge_v.emplace_back(trans_v_left[leftPos++]);
		}
		while (rightPos < trans_v_right.size()) {
			merge_v.emplace_back(trans_v_right[rightPos++]);
		}

		return merge_v;
	}
	/*========================================================================================*/

	/*======================     intersection 220818 10:13      =======================*/
	vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
		map<int, int> m;
		vector<int> res;
		std::unordered_set<int> s1(nums1.begin(), nums1.end());
		std::unordered_set<int> s2(nums2.begin(), nums2.end());

		for (auto au : s1) {
			m[au]++;
		}

		for (auto au : s2) {
			if (m[au]) {
				res.emplace_back(au);
			}
		}

		return res;
	}
	/*========================================================================================*/

	/*======================     Isomorphic Strings 220820 19:15      =======================*/
	bool isIsomorphic(string s, string t) {
		int n = s.size();
		map<char, int> m_s;
		map<char, int> m_t;

		for (int i = 0; i < n; i++) {
			m_s[s[i]]++;
			m_t[t[i]]++;
		}

		if (m_s[s[0]] != m_t[t[0]]) return false;

		for (int i = 1; i < n; i++) {
			if (m_s[s[i]] != m_t[t[i]]) return false;
			if (s[i] == s[i - 1] && t[i] != t[i - 1]) 
				return false;

			if (s[i] != s[i - 1] && t[i] == t[i - 1]) 
				return false;
		}

		return true;
	}
	/*========================================================================================*/

	/*===================     Minimum Index Sum of Two Lists 220821 19:53      =======================*/
	vector<string> findRestaurant(vector<string>& list1, vector<string>& list2) {
		map<string, int> m1;
		map<string, int> m2;
		vector<string> res;
		vector<string> r;

		for (int i = 0; i < list1.size(); i++) {
			m1[list1[i]] = i+1;
		}

		for (int i = 0; i < list2.size(); i++) {
			if (m1[list2[i]]) {
				m1[list2[i]] += i+1;
				res.emplace_back(list2[i]);
			}
		}

		sort(res.begin(), res.end(), [&m1](auto& a, auto& b) {return m1[a] < m1[b]; });

		r.emplace_back(res.front());

		for (int i = 1; i < res.size(); i++) {
			if (m1[r.back()] == m1[res[i]]) {
				r.emplace_back(res[i]);
			}
		}

		return r;
	}
	/*================================================================================================*/

	/*===================          Contains Duplicate II 220822 11:34          =======================*/
	bool containsNearbyDuplicate(vector<int>& nums, int k) {
		map<int,int> m;
		int n = nums.size();
		for (int i = 0; i < n; i++)
			if(m[nums[i]] && i  - m[nums[i]] < k) 
				return true;
			else 
				m[nums[i]] = i + 1;

		return false;
	}
	/*================================================================================================*/

	/*===================         Find Duplicate Subtrees 220823 12:03         =======================*/
	vector<TreeNode*> findDuplicateSubtrees(TreeNode* root) {
		vector<TreeNode*> res_v;
		unordered_map<string, int> m;

		DFS_findDuplicateSubtrees(root, m, res_v);

		return res_v;
	}

	string DFS_findDuplicateSubtrees(TreeNode* root, unordered_map<string, int>& m, vector<TreeNode*>& res_v) {
		if (!root) {
			return "";
		}

		string s = to_string(root->val) + "(" + DFS_findDuplicateSubtrees(root->left, m, res_v) + ")(" + DFS_findDuplicateSubtrees(root->right, m, res_v) + ")";

		if (m[s]++ == 1) {
			res_v.emplace_back(root);
		}

		return s;
	}
	/*================================================================================================*/

	/*================     Find Numbers with Even Number of Digits 220825 11:14      =================*/
	int findNumbers(vector<int>& nums) {
		int res = 0;

		for (auto& au : nums) {
			res += (int)log10(au) & 1;
		}

		return res;
	}
	/*================================================================================================*/

	/*=====================     Squares of a Sorted Array 220825 11:30      ==========================*/
	vector<int> sortedSquares(vector<int>& nums) {
		vector<int> res(nums.size());
		int l = 0, r = nums.size() - 1;
		for (int k = nums.size() - 1; k >= 0; k--) {
			if (abs(nums[r]) > abs(nums[l])) res[k] = nums[r] * nums[r--];
			else res[k] = nums[l] * nums[l++];
		}
		return res;
	}
	/*================================================================================================*/

	/*============================     Duplicate Zeros 220825 11:30      =============================*/
	void duplicateZeros(vector<int>& arr) {
		vector<int> res;
		int n = arr.size();
		for (auto& au : arr) {
			if (res.size() > n) {
				break;
			}
			if (au) {
				res.emplace_back(au);
			}
			else {
				res.emplace_back(0);
				res.emplace_back(0);
			}
		}
		while (res.size() > n) {
			res.pop_back();
		}
		arr.swap(res);
	}
	/*================================================================================================*/

	/*====================     Merge Sorted Array  220825 12:09  =====================================*/
	void merge_220825(vector<int>& nums1, int m, vector<int>& nums2, int n) {

		for (int i = 0; i < n; i++) {
			nums1.emplace_back(0);
		}

		int en = m + n;
		int i = m - 1;
		int j = n - 1;

		for (int en = m + n - 1; en >= 0 &&  j >= 0; en--) {
			if (i>=0 && nums1[i] >= nums2[j]) {
				nums1[en] = nums1[i--];
			}
			else {
				nums1[en] = nums2[j--];
			}
		}
	}
	/*================================================================================================*/

	/*====================  Check If N and Its Double Exist  220827 17:22  ===========================*/
	bool checkIfExist(vector<int>& arr) {
		map<int, int> m;
		bool flag = false;

		for_each(arr.begin(), arr.end(), [&m, &flag](int& au) {
			if (m[au * 2]) {
				flag = true;
				return;
			}

			if (!(au & 1) && m[au / 2]) {
				flag = true;
				return;
			}

			m[au]++;
			});

		return flag;
	}
	/*================================================================================================*/

	/*======================      Valid Mountain Array 220827 18:22      ============================*/
	bool validMountainArray(vector<int>& arr) {
		int n = arr.size();

		if (n < 3) 
			return false;

		int i = 0;
		while (i < n - 1 && arr[i] < arr[i+1] ) {
			i++;
		}

		int j = n - 1;
		while (j > 0 && arr[j - 1] > arr[j]) {
			j--;
		}

		return i == j && i > 0 && j < n-1;

	}
	/*================================================================================================*/

	/*============ Replace Elements with Greatest Element on Right Side 220828 14:04 =================*/
	vector<int> replaceElements(vector<int>& arr) {
		int m =-1;
		int n = arr.size() - 1;

		if (!n) return vector<int>{-1};

		for (int i = n; i >= 0; i--) {
			m = max(m, exchange(arr[i], m));
		}

		return arr;
	}
	/*================================================================================================*/

	/*==================================== Height Checker 220828 14:41 ===============================*/
	int heightChecker(vector<int>& heights) {
		int res = 0;
		int n = heights.size();
		vector<int> v(heights.begin(), heights.end());
		sort(v.begin(), v.end());
		
		for (int i = 0; i < n; i++) {
			if (heights[i] != v[i]) {
				res++;
			}
		}
		
		return res;
	}
	/*================================================================================================*/

	/*======================   Find All Numbers Disappeared in an Array 220828 14:41 =================*/
	vector<int> findDisappearedNumbers(vector<int>& nums) {
		vector<int> res;
		for (auto& au : nums) {
			if (nums[abs(au) - 1] > 0 ) {
				nums[abs(au) - 1] *= -1;
			}
		}

		for (int i = 0; i < nums.size(); i++) {
			if (nums[i]> 0) {
				res.emplace_back(i + 1);
			}
		}

		return res;
	}
	/*================================================================================================*/

	/*======================       498. Diagonal Traverse 220830 11:22     ===========================*/
	vector<int> findDiagonalOrder(vector<vector<int>>& mat) {
		int n = mat.size();
		int m = mat[0].size();
		vector<int> res;

		int r = 0;
		int c = 0;

		for (int i = 0; i < n + m; i++) {
			r = i;
			c = 0;
			vector<int> v;

			if (i >= n) {
				r = n - 1;
				c = i - n + 1;
			}

			while (r >= 0 && c < m)
				v.emplace_back(mat[r--][c++]);

			if (i & 1) 
				res.insert(res.end(), v.rbegin(), v.rend());
			else 
				res.insert(res.end(),v.begin(),v.end());
		}

		return res;
	}
	/*================================================================================================*/

	/*===========================       Add Binary 220830 14:53       ================================*/
	string addBinary(string a, string b) {
		bool one = false;

		int i = a.size();
		int j = b.size();
		if (i < j) {
			swap(a, b);
			swap(i, j);
		}

		while (--i >= 0 && --j >= 0) {
			if (a[i] == '1' && a[i] == b[j]) {
				a[i] = '0';
				if (one) 
					a[i] = '1';
				one = true;
			}
			else if (a[i] == '0' && a[i] == b[j]) {
				if (one) 
					a[i] = '1';
				one = false;
			}
			else if (a[i] == '0') {
				a[i] = '1';
				if (one) 
					a[i] = '0';
			}
			else if (a[i] == '1') {
				if (one) 
					a[i] = '0';
			}
		}

		while (i >= 0) {
			if (a[i] == '1') {
				if (one) 
					a[i] = '0';				
			}
			else if (a[i] == '0') {
				if (one) 
					a[i] = '1';
				one = false;
				break;
			}
			i--;
		}

		if (i < 0 && one) {
			a.insert(a.begin(), '1');
		}

		return a;
	}
	/*================================================================================================*/

	/*===========================    Minimum Size Subarray Sum 220902 12:48   ========================*/
	int minSubArrayLen(int target, vector<int>& nums) {
		int n = nums.size();
		int i = 0;
		int j = 0;
		//vector<int> dp(n, 0);
		int dp = INT_MAX;
		int sum = 0;
		while (i < n) {
			if (nums[i] >= target) return 1;
			sum += nums[i];
			while (sum >= target) {
				dp = min(i - j + 1, dp);
				sum -= nums[j++];
			}

			i++;
		}

		dp = dp != INT_MAX ? dp : 0;

		return dp;
	}
	/*================================================================================================*/
};
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/


/*&&&&&&&&&&&&&&&&&&&&&&&&&&&   Graph Node 220721 10:05       &&&&&&&&&&&&&&&&&&&&&&&&*/
// Definition for a Node.
class Node_Graph {
public:
	int val;
	vector<Node_Graph*> neighbors;
	Node_Graph() {
		val = 0;
		neighbors = vector<Node_Graph*>();
	}
	Node_Graph(int _val) {
		val = _val;
		neighbors = vector<Node_Graph*>();
	}
	Node_Graph(int _val, vector<Node_Graph*> _neighbors) {
		val = _val;
		neighbors = _neighbors;
	}

	void build_graph(vector<Node_Graph>& graph_vector, const vector<vector<int>>& od_vector) {
		int n = graph_vector.size();

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < od_vector[i].size();j++) {
				graph_vector[i].neighbors.emplace_back(&graph_vector[od_vector[i][j] - 1]);
			}
		}
	}

	Node_Graph* cloneGraph(Node_Graph* node) {
		if (!node) 
			return nullptr;
		else if (node->neighbors.empty()) 
			return new Node_Graph(node->val);

		map<Node_Graph*, Node_Graph*> node_map;
		Node_Graph* node_clone = new Node_Graph(node->val);
		node_map[node] = node_clone;
		clone_graph(node, node_clone, node_map);

		return node_clone;
	}

	void clone_graph(Node_Graph* node, Node_Graph* node_clone, map<Node_Graph*, Node_Graph*>& node_map) {

		
		for (auto au : node->neighbors) {
			if (!node_map[au]) {
				Node_Graph* node_clone_neighbor = new Node_Graph(au->val);
				node_map[au] = node_clone_neighbor;
				node_clone->neighbors.emplace_back(node_clone_neighbor);
			    clone_graph(au, node_clone_neighbor, node_map);
			}
			else node_clone->neighbors.emplace_back(node_map[au]);
		}
	}
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

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&     Design Circular Queue 220713 11：28    &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
class MyCircularQueue {
public:
	vector<int> q_vec;
	int pos_begin;
	int pos_end;
	int n;

	MyCircularQueue(int k) {
		q_vec.resize(k,INT_MIN);
		n = k;
		pos_begin = -1;
		pos_end = -1;
	}

	bool enQueue(int value) {
	
		if (isFull()) {
			return false;
		}
		else if (isEmpty()) {
			pos_begin = 0;
		}

		pos_end = (pos_end + 1) % n;
		q_vec[pos_end] = value;
		return true;
	}

	bool deQueue() {
		if (isEmpty()) {
			return false;
		}

		if (pos_begin == pos_end) {
			pos_begin = -1;
			pos_end = -1;
			return true;
		}

		pos_begin = (pos_begin + 1) % n;

		return true;
	}

	int Front() {
		if (isEmpty()) {
			return -1;
		}
		return q_vec[pos_begin];
	}

	int Rear() {
		if (isEmpty()) {
			return -1;
		}
		return q_vec[pos_end];
	}

	bool isEmpty() {
		return pos_begin == -1;
	}

	bool isFull() {
		if (n == 1) {
			if (q_vec[pos_begin] != INT_MIN)
				return true;
			else
				return false;
		}

		return (pos_end + 1) % n == pos_begin;
	}
};
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&       Min Stack 220718 10：33       &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
class MinStack {
public:

	stack<pair<int, int>> min_stack;

	MinStack() {

	}

	void push(int val) {
		if (min_stack.empty()) min_stack.emplace(val, val);
		else min_stack.emplace(val, min(val, min_stack.top().second));
	}

	void pop() {
		min_stack.pop();
	}

	int top() {
		return min_stack.top().first;
	}

	int getMin() {
		return min_stack.top().second;
	}
};
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&        My Queue 220725 13：20       &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

class MyQueue {
public:

	stack<int> s1;
	stack<int> s2;

	MyQueue() {
		
	}

	void push(int x) {
		s1.emplace(x);
	}

	int pop() {
		if (!s2.empty()) {
			int t = s2.top();
			s2.pop();
			return t;
		}

		while (!s1.empty()) {
			s2.emplace(s1.top());
			s1.pop();
		}

		int t = s2.top();
		s2.pop();
		return t;
	}

	int peek() {
		if (!s2.empty()) {
			return s2.top();
		}

		while (!s1.empty()) {
			s2.emplace(s1.top());
			s1.pop();
		}

		return s2.top();
	}

	bool empty() {
		return s1.empty() && s2.empty();
	}
};

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&        My Stack 220726 09：47       &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
class MyStack {
public:

	queue<int> q1;
	queue<int> q2;

	MyStack() {

	}

	void push(int x) {
		if (q1.empty()) {
			q2.emplace(x);
		}
		else {
			q1.emplace(x);
		}
	}

	int pop() {
		if (q1.empty()) {
			int n = q2.size();
			while (n-- > 1) {
				q1.emplace(q2.front());
				q2.pop();
			}
			int res = q2.front();
			q2.pop();
			return res;
		}
		else {
			int n = q1.size();
			while (n-- > 1) {
				q2.emplace(q1.front());
				q1.pop();
			}
			int res = q1.front();
			q1.pop();
			return res;
		}
	}

	int top() {
		if (q1.empty()) {
			int n = q2.size();
			while (n-- > 1) {
				q1.emplace(q2.front());
				q2.pop();
			}
			int res = q2.front();
			q1.emplace(q2.front());
			q2.pop();
			return res;
		}
		else {
			int n = q1.size();
			while (n-- > 1) {
				q2.emplace(q1.front());
				q1.pop();
			}
			int res = q1.front();
			q2.emplace(q1.front());
			q1.pop();
			return res;
		}
	}

	bool empty() {
		return q1.empty() && q2.empty();
	}
};

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack* obj = new MyStack();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->top();
 * bool param_4 = obj->empty();
 */
 /*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&        Design HashSet 220816 11:00       &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
class MyHashSet {
public:
	vector<bool> v;

	MyHashSet(): v(1e6 + 1, false) {
	}

	void add(int key) {
		v[key] = true;
	}

	void remove(int key) {
		v[key] = false;
	}

	bool contains(int key) {
		return v[key];
	}
};
/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/

/*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&        Design HashSet 220817 09:53       &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&*/
class MyHashMap {
public:

	vector<int> v;

	MyHashMap() : v(1e6,-1) {

	}

	void put(int key, int value) {
		v[key] = value;
	}

	int get(int key) {
		if (v[key] != -1) {
			return v[key];
		}
		return -1;
	}

	void remove(int key) {
		v[key] = -1;
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

/*====================     Two Sum      =======================*/
vector<int> Solution::twoSum_0(vector<int>& nums, int target) {
	map <int, pair<int, int>> mp;
	int i;
	vector <int> v;
	for (i = 0; i < nums.size(); i++)
		mp[nums[i]] = make_pair(1, i);

	for (i = 0; i < nums.size(); i++){
		if (mp[target - nums[i]].first > 0 && mp[target - nums[i]].second != i){
			v.push_back(i);
			v.push_back(mp[target - nums[i]].second);
			return v;
		}
	}
}
/*============================================================*/

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

string Solution::longestCommonPrefix_0(vector<string>& strs) {
	string prefix = "";
	for (int idx = 0; strs.size() > 0; prefix += strs[0][idx], idx++)
		for (int i = 0; i < strs.size(); i++)
			if (idx >= strs[i].size() || (i > 0 && strs[i][idx] != strs[i - 1][idx]))
				return prefix;
	return prefix;
}

//190105
bool Solution::isValid(string s) {
	stack<char> valid_stack;
	for (auto au : s) {
		if(au == '(' || au == '[' || au == '{') {
			valid_stack.emplace(au);
		}
		else if (!valid_stack.empty() && valid_stack.top() == '(' && au == ')') {
			valid_stack.pop();
		}
		else if (!valid_stack.empty() && valid_stack.top() == '[' && au == ']') {
			valid_stack.pop();
		}
		else if (!valid_stack.empty() && valid_stack.top() == '{' && au == '}') {
			valid_stack.pop();
		}
		else {
			return false;
		}
	}

	if (valid_stack.empty()) return true;

	return false;
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
/*====================     Remove Duplicates from Sorted Array  220826 10:46  =====================================*/
int Solution::removeDuplicates(vector<int>& nums) {

	//nums.erase(unique(nums.begin(), nums.end()), nums.end());
	//return nums.size();

	int n = nums.size();
	int i = 0;

	for (int j = 1; j < n; j++) 
		if (nums[i] != nums[j])
			nums[++i] = nums[j];
		
	return i+1;
}

//190108
/*====================     Remove Element  220826 10:32  =====================================*/
int Solution::removeElement(vector<int>& nums, int val) {
	/*int length = nums.size();
	int i = 0;
	while(i<length) {
		nums[i] == val ? nums[i] = nums[--length] : i++;
	}

	return length;*/

	/*while (find(nums.begin(), nums.end(), val) != nums.end())
		nums.erase(find(nums.begin(), nums.end(), val));
	return nums.size();*/

	int n = nums.size();
	int i = n - 1;
	int j = n;

	while (j-- > 0) {
		if (nums[j] == val) {
			swap(nums[j], nums[i--]);
		}
	}

	return i+1;
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

/*======================      Implement strStr() 220830 16:14      ============================*/
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

	return -1;

	return 0;*/

	int n = haystack.size();
	int	m = needle.size();

	if (!m) return 0;

	vector<int> lps = find_KMP(needle);

	for (int i = 0, j = 0; i < n;) {

		if (haystack[i] == needle[j]) {
			i++;
			j++;
		}

		if (j == m) {
			return i - j;
		}

		if (i < n && haystack[i] != needle[j]) {
			if (j) {
				j = lps[j - 1];
			}
			else {
				i++;
			}
		}
	}
	return -1;
}

vector<int> Solution::find_KMP(const string& const needle) {
	vector<int> lsp{0};
	int n = needle.size();

	int i = 1;
	int longest_prefix = 0;
	
	while (i < n) {
		if (needle[i] == needle[longest_prefix]) {
			lsp.emplace_back(++longest_prefix);
			i++;
		}
		else if (longest_prefix > 0) {
			longest_prefix = lsp[longest_prefix - 1];
		}
		else {
			lsp.emplace_back(0);
			i++;
		}
	}

	return lsp;
}

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

