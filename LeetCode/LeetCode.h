#pragma once
#include "pch.h"
#include <iostream>
#include <vector>
#include <map>
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
	ListNode* mergeTwoLists(ListNode *l1, ListNode *l2);
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
};

class MyLinkedList {
public:
	struct Node {
		int val=NULL;
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
		temp->next  =	new Node(val);
		temp->next->next = temp_node;
		size++;
	}

	/** Delete the index-th node in the linked list, if the index is valid. */
	void deleteAtIndex(int index) {
		if (index >= size) return;
		Node *temp = head;
		for (int i = 0; i < index; i++) temp = temp -> next;
		Node *temp_del = temp->next;
		temp->next = temp_del -> next;
		size--;
		delete temp_del;
		temp_del->next = nullptr;
	}
};

/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList obj = new MyLinkedList();
 * int param_1 = obj.get(index);
 * obj.addAtHead(val);
 * obj.addAtTail(val);
 * obj.addAtIndex(index,val);
 * obj.deleteAtIndex(index);
 */

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
	for(auto&& c:s)
		switch (c) {
		case'(': stk.push(')'); break;
		case'[': stk.push(']'); break;
		case'{': stk.push('}'); break;
		default:
			if(stk.empty()||c!=stk.top()) return false;
			else stk.pop();
		}
	return stk.empty();
}

//190106
struct Solution::ListNode {
	int val;
	Solution::ListNode *next;
	Solution::ListNode(int x) : val(x), next(nullptr) {}
};
Solution::ListNode* Solution::mergeTwoLists(Solution::ListNode *l1, Solution::ListNode *l2) {
	Solution::ListNode header(LONG_MIN);
	Solution::ListNode* tail_ptr=&header;

	while (l1&&l2) {
		Solution::ListNode** next_node = (l1->val<l2->val?&l1:&l2);
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
	nums.erase(unique(nums.begin(), nums.end()),nums.end());
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
		for (long long x = max; x*x >= u; x--)if (u%x == 0)return(int)(u % 1337);
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
	std::reverse(nums.begin(), nums.begin()+k);
	std::reverse(nums.begin() + k, nums.end());
	//2
	std::rotate(nums.begin(), nums.end() - k, nums.end());

}

//190121
string Solution::convertToTitle(int n) {
	return n == 0 ? "" : convertToTitle((n-1)/26)+(char)((n-1)%26+'A');
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
		else return firstBadVersionRec(Middle+1, R);
		
	}
}

//190221
int Solution::findPairs(vector<int>& nums, int k) {
	if (k < 0) return 0;
	int count = 0;
	unordered_multiset<int> ums(nums.begin(),nums.end());
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
		minrhs[i] = minr = min(minr,nums[i]);

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

	while (n-base*digits >0 ) {
		n -= base * digits;
		base *= 10;
		digits++;
	}

	int index = (n - 1) % digits;
	int offset = (n - 1) / digits;
	long start = pow(10,digits-1);
	return to_string(start + offset)[index] - '0';
}

//190224
uint32_t Solution::reverseBits(uint32_t n) {
	string str = bitset<32>(n).to_string();
	std::reverse(str.begin(),str.end());
	return (bitset<32>(str).to_ullong());
}


//190227
bool Solution::isPalindrome(string s) {
	if (s.size() <= 1) return true;

	auto left_end = s.begin();
	auto right_end = s.end();

	while (left_end < right_end) {
		while (!isalnum(*left_end)&&left_end<s.end()) left_end++;
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
	while (val*val>x) {
		val = (val+x/val) / 2;
	}

	return val;
}

//190304
bool Solution::canPlaceFlowers(vector<int>& flowerbed, int n) {
	flowerbed.insert(flowerbed.begin(),0);
	flowerbed.push_back(0);

	for (int i = 1; i < flowerbed.size() - 1; ++i) {
		if (flowerbed[i-1]+ flowerbed[i] + flowerbed[i+1]==0) {
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

	