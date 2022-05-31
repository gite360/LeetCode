// LeetCode.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "LeetCode.h"
#include <map>
#include <assert.h>

//class SortPolygon{
//public:

	

int main()
{

#if _DEBUG
	cout << "1 current id: " << 55555555555 << ", vector size: " << 555555555555 << endl;
#endif


	Solution solution;

	/*============== 34. Find First and Last Position of Element in Sorted Array 220531 ================*/
	vector<int> nums_34 = { 5,7,7,8,8,10 };
	int target_34 = 8;

	nums_34 = {5, 7, 7, 8, 8, 10};
	target_34 = 6;

	nums_34 = { 2,2 };
	target_34 = 1;

	solution.searchRange(nums_34, target_34);

	/*============== 322. Coin Change ================*/
	vector<int> coins_322{ 1 };
	coins_322 = { 2 };
	coins_322 = { 1,2,5 };
	coins_322 = { 186, 419, 83, 408 };
	

	int amount_322 = 1;
	amount_322 = 3;
	amount_322 = 11;
	amount_322 = 6249;

	solution.coinChange(coins_322, amount_322);

	/*========  172. Factorial Trailing Zeroes ========*/
	int n_172 = 10;
	solution.trailingZeroes(n_172);

	/*===============  334. Increasing Triplet Subsequence  ============*/
	vector<int> nums_334 = { 1, 2, 3, 4, 5 };
	nums_334 = { 5, 4, 3, 2, 1 };
	nums_334 = {2, 1, 5, 0, 4, 6};
	nums_334 = { 20,100,10,12,5,13 };
	//nums_334 = {6,7,1,2};
	//nums_334 = { 1,0,-1,0,10000 };
	solution.increasingTriplet(nums_334);
	/*===============   54. Spiral Matrix  ============*/
	vector<vector<int>> matrix_54 = { {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12} };
	matrix_54 = { {1,2,3} ,{4,5,6},{7,8,9} };
	matrix_54 = { {2, 3, 4}, {5, 6, 7}, {8, 9, 10}, {11, 12, 13}, {14, 15, 16} };
	solution.spiralOrder(matrix_54);

	/*====227. Basic Calculator II=======*/
	string s_227 = "1-1+1";

	//s_227 = " 3 / 2 ";
	//s_227 = " 3+5 / 2 ";
	s_227 = " 3+2*2";
	//s_227 = "0-2147483647";
	//s_227 = "1*2-3/4+5*6-7*8+9/10";
	s_227 = "     30";
	s_227 = "14/3/2";
	s_227 = "1+2*3+4*5+6*7+8*9+10";
	solution.calculate(s_227);

	/*===== 150. Evaluate Reverse Polish Notation =====*/
	vector<string> tokens = { "2", "1", "+", "3", "*" };
	//tokens = { "4","13","5","/","+" };
	tokens = { "10","6","9","3","+","-11","*","/","*","17","+","5","+" };
	//tokens = { "4","13","5","/","+" };
	solution.evalRPN(tokens);
	/*================================================*/

	/*======= 134. Gas Station =======*/
	vector<int> gas_134 = {1, 2, 3, 4, 5};
	vector<int> cost_134 = {3, 4, 5, 1, 2};

	gas_134 = { 2, 3, 4 };
	cost_134 = { 3, 4, 3 };

	solution.canCompleteCircuit(gas_134, cost_134);
	/*================================================*/

	/*======= 394. Decode String =======*/
	string s_394 = "sd2[f2[e]g]i";
	//string s_394 = "2[a]2[b]89[c]";
	//string s_394 = "2[l3[e4[c5[t]]]]";
	s_394 = "3[ab2[cd]e]fg10[h]";
	//string s_394 = "2[2[y]pq4[2[jk]e1[f]]]ef";
	//string s_394 = "3[a2[c]]";
	s_394 = "sd2[f2[e]g]i";
	solution.decodeString(s_394);
	/*================================================*/

	/*==== 114. Flatten Binary Tree to Linked List ====*/
	vector<Solution::TreeNode> node_vector_114 = { 1,2,5,3,4,-1,6 };
	vector<int> v_114 = { 1,2,5,3,4,-1,6 };
	Solution::TreeNode* root_114 = &node_vector_114[0];
	queue<Solution::TreeNode*> q_114;

	q_114.emplace(&node_vector_114[0]);
	int i = 0;
	//i++;
	while (!q_114.empty() && i < v_114.size()) {
		Solution::TreeNode* temp_node_114 = q_114.front();
		q_114.pop();
		i++;
		if (i < v_114.size() && v_114[i] != -1) {
			temp_node_114->left = &node_vector_114[i];
			q_114.emplace(temp_node_114->left);
		}
		i++;
		if (i < v_114.size() && v_114[i] != -1) {
			temp_node_114->right = &node_vector_114[i];
			q_114.emplace(temp_node_114->right);
		}

	}
		
	solution.flatten(root_114);
	/*================================================*/

	/*====96. Unique Binary Search Trees====*/
	int n_96 = 4;
	solution.numTrees(n_96);
	/*================================================*/

	/*===== 220521 24. Swap Nodes in Pairs ======*/
	
	Solution::ListNode* p = new Solution::ListNode(1);
	Solution::ListNode* head_24 = p;

	for (int i = 1; i < 5; i++) {
		Solution::ListNode* t = new Solution::ListNode(i+1);
		p->next = t;
		p = p->next;
	}

	solution.swapPairs(head_24);
	/*================================================*/

	/*===== 220520 64. Minimum Path Sum ======*/

	vector<vector<int>> grid{ {1,3,1}, {1,5,1}, {4,2,1} };

	solution.minPathSum(grid);
	/*================================================*/


	/*===== 220519 39. Combination Sum ======*/
	vector<int> candidates = { 2, 3, 6, 7 };
	int target_39 = 7;
	solution.combinationSum(candidates, target_39);
	/*================================================*/

	/*=== 220518 739. Daily Temperatures ====*/
	vector<int> temperatures{ 64,40,49,73,72,35,68,83,35,73,84,88,96,43,74,63,41,95,48,46,89,72,34,85,72,59,87,49,30,32,47,34,74,58,31,75,73,88,64,92,83,64,100,99,81,41,48,83,96,92,82,32,35,68,68,92,73,92,52,33,44,38,47,88,71,50,57,95,33,65,94,44,47,79,41,74,50,67,97,31,68,50,37,70,77,55,48,30,77,100,31,100,69,60,47,95,68,47,33,64 };
	solution.dailyTemperatures(temperatures);
	/*=================================*/
	string s_763 = "eccbbbbdec";
	solution.partitionLabels(s_763);
	/*=================================*/
	string s_139 = "catsandog";
	vector<string> wordDict{ "cats","dog","sand","and","cat" };

	solution.wordBreak(s_139, wordDict);
	assert(0);
	/*=================================*/
	string s_sub = "baaabcb"; 
	int k = 3;
	int rrrrr;
	int sssss = solution.longestSubstring(s_sub,k, rrrrr);
	cout << rrrrr << endl;

	/*=================================*/
	vector<vector<int>> intervals = { {1, 4}, { 0, 2 },  { 3, 5 } };// { {1, 4}, { 0, 2 },  { 3, 5 } }; // { {4, 5}, { 1, 4 } }; // { {1, 3}, { 2, 6 }, { 8, 10 }, { 15, 18 } };
	solution.merge(intervals);

	/*===============================*/
	int numCourses = 4;
	vector<vector<int>> prerequisites = { {1, 0}, {2, 0}, {3, 1}, {3, 2} };//{ {1, 0}, {2, 0}, {3, 1}, {3, 2} };
	solution.canFinish(numCourses, prerequisites);

	/*===============================*/
	vector<int> nums_peak = { 1,2};
	solution.findPeakElement(nums_peak);

	/*===============================*/
	
	solution.findOrder(numCourses, prerequisites);

	/*===============================*/
	//vector<int> job_nums = { 226,174,214,16,218,48,153,131,128,17,157,142,88,43,37,157,43,221,191,68,206,23,225,82,54,118,111,46,80,49,245,63,25,194,72,80,143,55,209,18,55,122,65,66,177,101,63,201,172,130,103,225,142,46,86,185,62,138,212,192,125,77,223,188,99,228,90,25,193,211,84,239,119,234,85,83,123,120,131,203,219,10,82,35,120,180,249,106,37,169,225,54,103,55,166,124 };
	vector<int> job_nums = { 6,3,10,8,2,10,3,5,10,5,3 };
	//vector<int> job_nums = { 4,1,2,7,5,3,1 };
	solution.rob(job_nums);

	/*===============================*/
	vector<int> height = { 4,2,0,3,2,5 };
	solution.trap(height);

	/*===============================*/

	//int N = 1000;
	//int m = 3;
	//cin >> N >> m;
	////vector<vector<int>> bag;
	//vector<vector<int>>dp(m + 1, vector<int>(N + 1, 0));
	//vector<vector<int>> price(61, vector<int>(3, 0));
	//vector<vector<int>> priority(61, vector<int>(3, 0));

	//int a, b, c;

	//for (int i = 1; i <= m; i++) {
	//	cin >> a >> b >> c;
	//	if (c == 0) {
	//		price[i][0] = a;
	//		priority[i][0] = b;
	//	}
	//	else if (price[c][1] == 0) {
	//		price[c][1] = a;
	//		priority[c][1] = b;
	//	}
	//	else {
	//		price[c][2] = a;
	//		priority[c][2] = b;
	//	}
	//}

	solution.huawei0();

	/*===============================*/
	//int result_h = 0;
	//int a[6][2] = { {1000,5}, {800,2}, {400,5}, {300,5}, {400,3} , {200,2} };
	//int* p = a[0];
	//solution.GetResult(p, result_h);
	/*================================*/
	typename Solution::Node138 head0(7);
	typename Solution::Node138 head1(13);
	typename Solution::Node138 head2(11);
	typename Solution::Node138 head3(10);
	typename Solution::Node138 head4(1);
	head0.next = &head1;
	head1.next = &head2;
	head1.random = &head0;
	head2.next = &head3;
	head2.random = &head4;
	head3.next = &head4;
	head3.random = &head2;
	head4.next = nullptr;
	head4.random = &head0;


	solution.copyRandomList(&head0);

	/*==================================*/
	vector<vector<int>> matrix = {{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
	solution.setZeroes(matrix);


	/*==================================*/
	string result = solution.countAndSay(4);

	assert(0);

	vector<int> nums = { 1,1,2 };
	int target = 0;
	bool modified = false;
	string s = "()";
	//solution.twoSum(nums, target);

	vector<string> strs = { "flower","flow","flight" };
	cout << strs.size() << " "<< strs[1].size() << endl;

	string haystack = "a";
	string needle = "a";
	solution.strStr(haystack, needle);
	assert(0);

	int n = 27;
	solution.isPowerOfThree(n);
	assert(0);

	if (solution.isValid(s)) {
		cout << "true" << endl;
	}
	else
		cout << "false" << endl;

	//cout << solution.removeDuplicates(nums) << endl;
	//cout<< solution.removeElement(nums, 1);

	//if (++modified) cout << "aaa" << endl;
	//else cout << "bbb" << endl;

	cout << target <<"! int : "<< !target << endl;

	cout << solution.findNthDigit(5) << endl; 
	cout << solution.convertToTitle(19) << endl;
	cout << solution.convertToTitle(43) << endl;


	multimap<int, char> multiple_map{ {0,'a'},{1,'b'},{1,'c'},{1,'d'},{2,'e'},{3,'f'} };

	auto density_nodeHandler = multiple_map.extract(1);
	density_nodeHandler.key() = 2;
	multiple_map.insert(std::move(density_nodeHandler));
	//density_nodeHandler.key() = getMergedSubSegmentInformation(splited_node);
	//merge_segment_density_map.insert(std::move(density_nodeHandler));

	
	system("pause");

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
