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
	bool modified = false;
	for (int i = 1; i < nums.size();i++) {
		if (nums[i - 1] > nums[i]) {
			if (modified++) return false;
			i - 2 < 0 || nums[i - 2] <= nums[i] ? nums[i - 1] = nums[i] : nums[i] = nums[i - 1];
		}

	}
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



	