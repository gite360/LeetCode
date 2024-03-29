﻿// LeetCode.cpp : This file contains the 'main' function. Program execution begins and ends there.
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
	Node_Graph node_graph;

	int k;
	int val;
	int num;
	int target;

	vector<int> inorder;
	vector<int> preorder;
	vector<int> postorder;
	vector<int> nums;
	vector<int> arr;
	vector<vector<int>> mat;
	vector<vector<int>> queries;
	

	char target_char;
	vector<char> letters_char_v;
	string sentence;
	string haystack;
	string needle;
	string s;
	vector<string> strs;
	vector<string> nums_str;
	vector<string> dictionary;

	Solution::TreeNode* root = nullptr;
	vector<Solution::TreeNode> node_vector;

	/*=======================       Replace Words 221018 11:46      ==================================*/
	dictionary = { "cat","bat","rat" };
	sentence = "the cattle was rattled by the battery";
	solution.replaceWords(dictionary, sentence);

	/*===================       Maximum Depth of N-ary Tree 2201004 13:27      =======================*/
	node_vector = { 1,INT_MIN,3,2,4,INT_MIN,5,6 };
	solution.build_Nary_tree_level_order(node_vector, root);
	root = &node_vector[0];
	int test = solution.maxDepth(root);
	cout << test;
	assert(0);
	/*=====================     N-ary Tree Preorder Traversal 2201001 20:15     ======================*/
	node_vector = { 1,INT_MIN,3,2,4,INT_MIN,5,6 };
	root = &node_vector[0];
	solution.preorder(root);

	/*=========================     Balanced Binary Tree 220929 11:53     ============================*/
	node_vector = { 1, 2, 2, 3, 3, INT_MIN, INT_MIN, 4, 4 };
	root = &node_vector[0];
	solution.build_b_tree_level_order(node_vector, root);

	solution.isBalanced(root);

	/*====================     Insert into a Binary Search Tree 220924 12:02     =====================*/
	val = 5;
	solution.insertIntoBST(root, val);

	/*====================  Find Smallest Letter Greater Than Target 220916 14:47  ===================*/
	node_vector = { 7, 3, 15, INT_MIN, INT_MIN, 9, 20};
	root = &node_vector[0];
	solution.build_b_tree_level_order(node_vector, root);

	BSTIterator* obj_BSTIterator = new BSTIterator(root);

	/*====================  Find Smallest Letter Greater Than Target 220916 14:47  ===================*/
	assert(0);
	letters_char_v = { 'c', 'f', 'j' };
	target_char = 'a';
	target_char = 'c';
	solution.nextGreatestLetter(letters_char_v, target_char);

	/*=======================        Valid Perfect Square 220915 10:34         ==============================*/
	num = 16;
	num = 14;
	num = 5;
	solution.isPerfectSquare(num);

	/*=======================        Binary Search 220909 13:14         ==============================*/
	nums = {2,5};
	target = 5;
	solution.search_220909(nums, target);

	/*===================    Query Kth Smallest Trimmed Number 220907 17:52   ========================*/
	nums_str = { "102", "473", "251", "814" };
	queries = { {1, 1}, {2, 3}, {4, 2}, {1, 2}};

	solution.smallestTrimmedNumbers(nums_str, queries);

	/*=======================    Reverse Words in a String III 220904 13:23   ========================*/
	s = "Let's take LeetCode contest";
	s = "hehhhhhhe";

	solution.reverseWords_3(s);

	/*===========================    Reverse Words in a String 220903 19:23   ========================*/
	s = "  hello world  ";
	solution.reverseWords(s);

	/*===========================    Minimum Size Subarray Sum 220902 12:48   ========================*/
	target = 7;
	nums = { 2,3,1,2,4,3 };
	solution.minSubArrayLen(target, nums);

	/*==================     220901 11:10 Two Sum II - Input array is sorted     ================*/
	assert(0);
	nums = { 2,7,11,15 };
	target = 9;

	solution.twoSum(nums, target);

	/*==================     220831 12:06 Longest Common Prefix     ================*/
	strs = { "flower", "flow", "flight" };

	solution.longestCommonPrefix(strs);

	/*======================      Implement strStr() 220830 16:14      ============================*/

	haystack = "AABAACAADAABAABA";
	needle = "BAAB";

	haystack = "ABABABCABABABCABABABC";
	needle = "ABABAC";

	solution.strStr(haystack, needle);

	/*==============================      Add Binary 220830 15:19     ================================*/
	string addBinary_a;
	string addBinary_b;

	addBinary_a = "1010";
	addBinary_b = "1011";

	solution.addBinary(addBinary_a, addBinary_b);

	/*==============================   498. Diagonal Traverse 220830 11:22   =========================*/
	assert(0);
	mat = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
	solution.findDiagonalOrder(mat);

	/*======================   Find All Numbers Disappeared in an Array 220828 14:41 =================*/
	assert(0);
	nums = { 4,3,2,7,8,2,3,1 };
	solution.findDisappearedNumbers(nums);

	/*======================      Valid Mountain Array  220827 18:22      ============================*/
	assert(0);
	arr = { 0,3,2,1 };
	arr = { 3,5,5};
	//arr = { 2,0,2 };

	solution.validMountainArray(arr);

	/*====================  Check If N and Its Double Exist  220827 17:22  ===========================*/
	assert(0);
	arr = { 10,2,5,3 };
	arr = { -16, -13, 8};
	arr = {-10,12,-20,-8,15};

	solution.checkIfExist(arr);

	/*====================     Remove Duplicates from Sorted Array  220826 10:46  =====================================*/
	assert(0);
	nums = { 0,0,1,1,1,2,2,3,3,4 };
	//nums = { 1,1 };

	solution.removeDuplicates(nums);

	/*====================     Remove Element  220826 10:32  =====================================*/
	assert(0);
	nums = { 3,2,2,3 };
	k = 3;
	solution.removeElement(nums, k);

	/*====================     Merge Sorted Array  220825 12:09  =====================================*/
	assert(0);
	vector<int> nums1_220825;
	int m_220825;
	vector<int> nums2_220825;
	int n_220825;

	nums1_220825 = { 1,2,3 };
	nums2_220825 = { 2,5,6 };

	nums1_220825 = {};
	nums2_220825 = { 1 };

	m_220825 = nums1_220825.size();
	n_220825 = nums2_220825.size();
	solution.merge_220825(nums1_220825, m_220825, nums2_220825, n_220825);

	/*================     Find Numbers with Even Number of Digits 220825 11:14      =================*/
	assert(0);
	vector<int> nums_findNumbers;

	nums_findNumbers = { 12,345,2,6,7896 };

	solution.findNumbers(nums_findNumbers);

	/*==== 3. Longest Substring Without Repeating Characters ======*/
	assert(0);
	string s_3 = "wobgrovw";
	s_3 = "dvdf";
	s_3 = "abcabcbb";
	s_3 = "pwwkew";
	s_3 = "aab";

	solution.lengthOfLongestSubstring(s_3);

	/*===================          Contains Duplicate II 220822 11:34          =======================*/
	vector<int> nums_containsNearbyDuplicate;
	int k_containsNearbyDuplicate;

	nums_containsNearbyDuplicate = {99, 99};
	k_containsNearbyDuplicate = 2;

	solution.containsNearbyDuplicate(nums_containsNearbyDuplicate, k_containsNearbyDuplicate);

	/*===================     Minimum Index Sum of Two Lists 220821 19:53      =======================*/
	vector<string> list1_findRestaurant;
	vector<string> list2_findRestaurant;
	list1_findRestaurant = { "Shogun","Tapioca Express","Burger King","KFC" };
	list2_findRestaurant = { "KFC","Shogun","Burger King" };

	solution.findRestaurant(list1_findRestaurant, list2_findRestaurant);

	/*======================     Isomorphic Strings 220820 19:15      =======================*/
	string s_Isomorphic = "egg";
	string t_Isomorphic = "add";
	solution.isIsomorphic(s_Isomorphic, t_Isomorphic);

	/*======================     The Skyline Problem 220815 10:28      =======================*/
	vector<vector<int>> buildings_skyline;
	buildings_skyline = {{2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}};

	solution.getSkyline(buildings_skyline);

	/*===========     Letter Combinations of a Phone Number 220814 19:25      ================*/
	assert(0);
	string digits_letterCombinations;
	digits_letterCombinations = "23";

	solution.letterCombinations(digits_letterCombinations);

	/*========================      Permutations 220812 15:23      ===========================*/
	assert(0);
	vector<int> nums_permute;
	nums_permute = { 1,2,3 };

	solution.permute(nums_permute);

	/*==============      Largest Rectangle in Histogram 220811 10:31      ===================*/
	assert(0);
	vector<int> heights_largestRectangleArea;

	//heights_largestRectangleArea = {2, 1, 5, 6, 2, 3};
	//heights_largestRectangleArea = { 0 };
	//heights_largestRectangleArea = { 4,2 };
	//heights_largestRectangleArea = { 2, 4 };
	//heights_largestRectangleArea = { 0,0 };
	//heights_largestRectangleArea = { 0,9 };
	//heights_largestRectangleArea = { 0,2,0 };
	heights_largestRectangleArea = { 2, 2, 6, 7, 5, 5, 5, 0, 9, 3, 6, 3, 8, 6, 6 };
	//heights_largestRectangleArea = { 9,0 };
	//heights_largestRectangleArea = { 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9 };
	//heights_largestRectangleArea = { 2, 2, 6, 6, 5, 5, 5 };
	heights_largestRectangleArea = { 9, 8, 7, 6, 5, 6, 7, 8, 9, 10, 11};

	heights_largestRectangleArea = { 1,2,3 };
	solution.largestRectangleArea(heights_largestRectangleArea);

	/*======================      generateParenthesis 220810 11:35      ============================*/
	assert(0);
	int n_generateParenthesis;
	n_generateParenthesis = 3;

	solution.generateParenthesis(n_generateParenthesis);

	/*======================      Combinations 220809 10:35      ============================*/
	assert(0);
	int n_Combinations = 4;
	int k__Combinations = 3;

	solution.combine(n_Combinations, k__Combinations);

	/*======================      Sudoku Solver 220808 12:22      ============================*/
	assert(0);
	vector<vector<char>> board_solveSudoku;
	board_solveSudoku = {{'5', '3', '.', '.', '7', '.', '.', '.', '.'}, 
		{'6', '.', '.', '1', '9', '5', '.', '.', '.'}, 
		{'.', '9', '8', '.', '.', '.', '.', '6', '.'}, 
		{'8', '.', '.', '.', '6', '.', '.', '.', '3'}, 
		{'4', '.', '.', '8', '.', '3', '.', '.', '1'}, 
		{'7', '.', '.', '.', '2', '.', '.', '.', '6'}, 
		{'.', '6', '.', '.', '.', '.', '2', '8', '.'}, 
		{'.', '.', '.', '4', '1', '9', '.', '.', '5'}, 
		{'.', '.', '.', '.', '8', '.', '.', '7', '9'}};

	solution.solveSudoku(board_solveSudoku);

	/*======================      N-Queens II 220807 15:14        ============================*/
	assert(0);
	int n_totalNQueens = 4;
	solution.totalNQueens(n_totalNQueens);

	/*======================     Sort an Array 220805 18:48       ============================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_98;

	node_vector_98 = { 5,1,4,INT_MIN,INT_MIN,3,6 };

	node_vector_98 = { 2,1,3 };

	node_vector_98 = { 5,4,6,INT_MIN,INT_MIN,3,7 };

	node_vector_98 = { 1,1 };

	//node_vector_98 = { 1,0 };

	Solution::TreeNode* root_98 = &node_vector_98[0];

	solution.build_b_tree_level_order(node_vector_98, root_98);

	solution.isValidBST(root_98);

	/*======================     Sort an Array 220805 18:48       ============================*/
	assert(0);
	vector<int> nums_sortArray = { 5, 2, 3, 1 };
	//nums_sortArray = {5, 1, 1, 2, 0, 0, -6};
	solution.sortArray(nums_sortArray);

	/*===============     Unique Binary Search Trees II 220805 12:35       ===================*/
	assert(0);
	int n_generateTrees = 3;
	solution.generateTrees(n_generateTrees);

	/*===================        K-th Symbol in Grammar 220804 13:50       ===================*/
	assert(0);
	int n_kthGrammar = 2;
	int k_kthGrammar = 2;

	n_kthGrammar = 30;
	k_kthGrammar = 434991989;

	/*n_kthGrammar = 3;
	k_kthGrammar = 3;*/

	solution.kthGrammar(n_kthGrammar, k_kthGrammar);

	/*=================         Pascal's Triangle II 220802 12:13         =======================*/
	assert(0);
	int rowIndex_pascal_triangle = 3;
	solution.getRow(rowIndex_pascal_triangle);

	/*=======================         Keys and Rooms 220801 09:55         =======================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_searchBST;
	node_vector_searchBST = { 4, 2, 7, 1, 3 };
	int val_searchBST = 2;
	val_searchBST = 5;

	node_vector_searchBST = { 18,2,22, INT_MIN , INT_MIN , INT_MIN ,63, INT_MIN ,84, INT_MIN , INT_MIN};
	val_searchBST = 63;

	Solution::TreeNode* root_searchBST = &node_vector_searchBST[0];

	solution.build_b_tree_level_order(node_vector_searchBST, root_searchBST);
	solution.searchBST(root_searchBST, val_searchBST);

	/*=======================         Keys and Rooms 220730 18:45         =======================*/
	assert(0);
	vector<vector<int>> rooms_canVisitAllRooms;

	rooms_canVisitAllRooms = { {1}, {2}, {3}, {} };

	rooms_canVisitAllRooms = {{1, 3}, {3, 0, 1}, {2}, {0}};
	/*rooms_canVisitAllRooms = {{1, 7, 9}, {8, 3, 6}, {1}, {6, 5}, {4, 7}, {5, 2, 6}, {4, 5}, {2}, {9, 8, 2, 3, 4}, {1, 3, 9}};

	rooms_canVisitAllRooms = {{10, 35, 25, 33}, {}, {47, 27}, {23, 3, 28, 39}, {36, 41, 10, 24}, {14, 40, 42, 44}, {41, 35}, {48}, {24, 26, 44, 23, 11, 17}, {1, 17, 20, 30}, {},
		{38, 39, 44}, {}, {11, 21, 45, 13}, {1, 27, 14, 19}, {23}, {30, 21, 8, 22, 40}, {14, 48}, {32, 6}, {5, 15, 26, 34, 38}, {43, 25}, {18, 10, 33}, {15, 34, 9},
		{18, 46, 48, 7, 13}, {5, 29, 4}, {42, 34}, {7, 13, 37, 8, 15, 21}, {4, 5}, {38, 20, 42}, {16, 19, 47}, {8, 29}, {28, 33, 37, 49}, {9, 39, 49, 41},
		{31, 12, 26, 32}, {2, 40, 32, 46}, {27, 22}, {12, 37, 2}, {31, 1}, {46, 19, 16, 18, 30}, {35}, {6, 3, 7, 28, 43}, {4, 25, 2, 29}, {},
		{22, 24, 45, 12}, {36, 31}, {3, 36, 45}, {20, 43, 49}, {11}, {16, 6}, {17, 47}};*/

	solution.canVisitAllRooms(rooms_canVisitAllRooms);

	/*=======================         01 Matrix 220729 10:31         =======================*/
	assert(0);
	vector<vector<int>> mat_updateMatrix;

	mat_updateMatrix = 
	{{0, 0, 0}, 
	{0, 1, 0}, 
	{1, 1, 1}};

	/*mat_updateMatrix = {{1, 0, 1, 1, 0, 0, 1, 0, 0, 1}, 
		{0, 1, 1, 0, 1, 0, 1, 0, 1, 1}, 
		{0, 0, 1, 0, 1, 0, 0, 1, 0, 0}, 
		{1, 0, 1, 0, 1, 1, 1, 1, 1, 1}, 
		{0, 1, 0, 1, 1, 0, 0, 0, 0, 1}, 
		{0, 0, 1, 0, 1, 1, 1, 0, 1, 0}, 
		{0, 1, 0, 1, 0, 1, 0, 0, 1, 1}, 
		{1, 0, 0, 0, 1, 1, 1, 1, 0, 1}, 
		{1, 1, 1, 1, 1, 1, 1, 0, 1, 0}, 
		{1, 1, 1, 1, 0, 1, 0, 0, 1, 1}};

	mat_updateMatrix = { {0, 0}, {0, 1}};

	mat_updateMatrix = { {0, 0, 0}, 
		{0, 0, 1}, 
		{0, 1, 1} };*/

	mat_updateMatrix = {{1, 1, 0, 0, 1, 0, 0, 1, 1, 0},
		{1, 0, 0, 1, 0, 1, 1, 1, 1, 1},
		{1, 1, 1, 0, 0, 1, 1, 1, 1, 0},
		{0, 1, 1, 1, 0, 1, 1, 1, 1, 1},
		{0, 0, 1, 1, 1, 1, 1, 1, 1, 0},
		{1, 1, 1, 1, 1, 1, 0, 1, 1, 1},
		{0, 1, 1, 1, 1, 1, 1, 0, 0, 1},
		{1, 1, 1, 1, 1, 0, 0, 1, 1, 1},
		{0, 1, 0, 1, 1, 0, 1, 1, 1, 1},
		{1, 1, 1, 0, 1, 0, 1, 1, 1, 1}};

	solution.updateMatrix_0(mat_updateMatrix);

	/*==============    394. Decode String 22/07/27 08:48     ==========*/
	string s_decodeString;

	s_decodeString = "2[abc]3[cd]ef";
	//s_decodeString = "3[a2[c]]";

	solution.decodeString_1(s_decodeString);

	/*=======================         Target Sum 220722 08:41       =======================*/
	assert(0);
	vector<int> nums_target_sum_vector;
	int	target_target_sum;

	nums_target_sum_vector = { 1,1,1,1,1 };
	target_target_sum = 3;

	solution.findTargetSumWays(nums_target_sum_vector, target_target_sum);

	/*======== Graph Node 220721 10:05 =======*/
	assert(0);
	vector<Node_Graph> node_graph_vecor;
	vector<vector<int>> od_node_graph_vector;

	node_graph_vecor = {1,2,3,4};
	od_node_graph_vector = {{2, 4}, {1, 3}, {2, 4}, {1, 3}};
	Node_Graph* node_graph_pointer = &node_graph_vecor[0];

	node_graph.build_graph(node_graph_vecor, od_node_graph_vector);

	node_graph.cloneGraph(node_graph_pointer);
	
	/*========   150. Evaluate Reverse Polish Notation  220720  08:29   =======*/
	assert(0);
	vector<vector<char>> grid_numIsland;

	grid_numIsland = {
			{'1', '1', '1', '1', '0'},
			{'1', '1', '0', '1', '0'},
			{'1', '1', '0', '0', '0'},
			{'0', '0', '0', '0', '0'}
	};

	grid_numIsland = {
		{'1', '1', '0', '0', '0'},
			{'1', '1', '0', '0', '0'},
			{'0', '0', '1', '0', '0'},
			{'0', '0', '0', '1', '1'}
	};
	solution.numIslands(grid_numIsland);
	/*========   150. Evaluate Reverse Polish Notation  220720  08:29   =======*/
	assert(0);
	vector<string> tokens_polish_notation;
	tokens_polish_notation = { "10","6","9","3","+","-11","*","/","*","17","+","5","+" };
	solution.evalRPN(tokens_polish_notation);

	/*==================     220719 39. Daily Temperatures     ================*/
	assert(0);
	vector<int> temperatures_daily;
	temperatures_daily = { 73,74,75,71,69,72,76,73 };
	solution.dailyTemperatures(temperatures_daily);
	/*===================        MinStack minStack     ========================*/
	assert(0);
	MinStack* minStack = new MinStack();
	minStack->push(-2);
	minStack->push(0);
	minStack->push(-3);
	minStack->getMin(); // return -3
	minStack->pop();
	minStack->top();    // return 0
	minStack->getMin(); // return -2

	/*=======================         Perfect Squares 220717 19:15       =======================*/
	assert(0);
	int n_perfect_squares;
	n_perfect_squares = 12;
	n_perfect_squares = 13;

	solution.numSquares(n_perfect_squares);

	/*=======================         Open the Lock 220716 19:07       =======================*/
	assert(0);
	vector<string> deadends_lock ;
	string target_lock = "0202";

	deadends_lock = { "0201", "0101", "0102", "1212", "2002" };
	target_lock = "0202";

	deadends_lock = { "8888" };
	target_lock = "0009";

	deadends_lock = { "8887","8889","8878","8898","8788","8988","7888","9888" };
	target_lock = "8888";

	solution.openLock(deadends_lock, target_lock);

	/*=======================     Number of Islands 220714 11:57     =========================*/
	assert(0);

	solution.numIslands(grid_numIsland);

	/*============     Serialize and Deserialize Binary Tree 220712 10:51     ================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_serialize;
	node_vector_serialize = { 1,2,3,INT_MIN,INT_MIN,4,5};
	//node_vector_serialize = { 1,2,3,4,5,INT_MIN,7 };
	Solution::TreeNode* root_serialize = &node_vector_serialize[0];
	solution.build_b_tree_level_order(node_vector_serialize, root_serialize);

	solution.deserialize(solution.serialize(root_serialize));

	/*===========      Lowest Common Ancestor of a Binary Tree 220711 10:36     ==============*/
	/*=========     Populating Next Right Pointers in Each Node 220707 13:23     =============*/
	assert(0);
	vector<Solution::TreeNode> node_vector_populate;
	node_vector_populate = { 1,2,3,4,5,6,7 };
	node_vector_populate = { 1,2,3,4,5,INT_MIN,7 };
	Solution::TreeNode* root_populate = &node_vector_populate[0];
	solution.build_b_tree_level_order(node_vector_populate, root_populate);
	solution.connect_next_right(root_populate);

	/*==== Construct Binary Tree from Preorder and Inorder Traversal 220706 10:39 ===========*/
	assert(0);
	preorder = { 3,9,20,15,7 };;
	inorder = { 9,3,15,20,7 };

	solution.buildTree_preorder_inorder(preorder, inorder);

	/*=========================    Path Sum  220705 10:28    ===================================*/
	assert(0);
	inorder = { 9,3,15,20,7 };
	postorder = { 9,15,7,20,3 };

	//inorder = { 2,1 };
	//postorder = { 2,1 };

	//inorder = { 1,2,3,4 };
	//postorder = { 1,4,3,2 };

	solution.buildTree_inorder_postorder(inorder, postorder);

	/*=========================    Path Sum  220704 10:06    ===================================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_PathSum;
	int targetSum_PathSum = 22;
	node_vector_PathSum = { 1,2,3,4,5 };
	//node_vector_level = { 1,2,3 };

	node_vector_PathSum = { 5,4,8,11,INT_MIN,13,4,7,2,INT_MIN,INT_MIN,INT_MIN,1 };
	targetSum_PathSum = 22;

	node_vector_PathSum = {1, 2, INT_MIN, 3, INT_MIN, 4, INT_MIN, 5};
	targetSum_PathSum = 6;

	Solution::TreeNode* root_PathSum = &node_vector_PathSum[0];
	solution.build_b_tree_level_order(node_vector_PathSum, root_PathSum);

	solution.hasPathSum(root_PathSum, targetSum_PathSum);

	/*=========================    Symmetric Tree 220704 09:41    ==============================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_symmetric;
	node_vector_symmetric = { 1,2,3,4,5 };
	//node_vector_level = { 1,2,3 };
	Solution::TreeNode* root_symmetric = &node_vector_symmetric[0];
	solution.build_b_tree_level_order(node_vector_symmetric, root_symmetric);

	solution.isSymmetric(root_symmetric);

	/*================     Binary Tree Level Order Traversal  220701 16:16    ==================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_level;
	node_vector_level = { 1,2,3,4,5 };
	//node_vector_level = { 1,2,3 };
	Solution::TreeNode* root_level = &node_vector_level[0];
	solution.build_b_tree_level_order(node_vector_level, root_level);

	solution.levelOrder(root_level);

	/*================     Binary Tree Preorder Traversal 220630 11：13    ==================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_preorder;

	node_vector_preorder = { 1,2,3,4,5 };

	//node_vector_543 = { 1,2,3 };

	Solution::TreeNode* root_preorder = &node_vector_preorder[0];

	solution.build_b_tree_level_order(node_vector_preorder, root_preorder);

	solution.preorderTraversal(root_preorder);

	/*=============     35. Search Insert Position 220629 09：30    =======================*/
	assert(0);
	vector<int> nums_35;
	int target_35;

	nums_35 = { 1,3,5,6 };
	target_35 = 5;

	/*nums_35 = { 1,3,5,6 };
	target_35 = 2;*/

	/*nums_35 = { 1,3};
	target_35 = 3;*/

	nums_35 = { 1,3,5};
	target_35 = 4;

	nums_35 = { 3,5,7,9,10 };
	target_35 = 8;

	solution.searchInsert(nums_35, target_35);

	/*===========     543. Diameter of Binary Tree 220628 10：25    =======================*/
	vector<Solution::TreeNode> node_vector_543;

	node_vector_543 = { 1,2,3,4,5 };

	//node_vector_543 = { 1,2,3 };

	Solution::TreeNode* root_543 = &node_vector_543[0];

	solution.build_b_tree_level_order(node_vector_543, root_543);

	solution.diameterOfBinaryTree(root_543);

	/*=====================     226. Invert Binary Tree 10:29    ==========================*/
	assert(0);
	vector<Solution::TreeNode> node_vector_226;

	node_vector_226 = { 4,2,7,1,3,6,9 };

	Solution::TreeNode* root_226 = &node_vector_226[0];

	solution.build_b_tree_level_order(node_vector_226, root_226);

	solution.invertTree(root_226);

	/*=======  338. Counting Bits  =========*/
	assert(0);
	int n_338=0;

	solution.countBits(n_338);

	/*======   91. Decode Ways 220624 15:14 ===========*/
	assert(0);
	string s_91;

	s_91 = "12";

	s_91 = "226";

	//s_91 = "06";

	//s_91 = "1111111111111";

	s_91 = "2101";

	solution.numDecodings(s_91);

	/*=======     98. Validate Binary Search Tree   =========*/
	assert(0);
	/*vector<Solution::TreeNode> node_vector_98;

	node_vector_98 = { 5,1,4,INT_MIN,INT_MIN,3,6 };

	node_vector_98 = {2,1,3};

	node_vector_98 = { 5,4,6,INT_MIN,INT_MIN,3,7 };

	node_vector_98 = { 1,1 };

	node_vector_98 = { 1,0 };

	Solution::TreeNode* root_98 = &node_vector_98[0];

	solution.build_b_tree_level_order(node_vector_98, root_98);

	solution.isValidBST(root_98);*/

	//node_vector_98.clear();
	//node_vector_98.shrink_to_fit();

	/*=========================       15. 3Sum      =======================*/
	vector<int> nums_15;

	nums_15 = { -1,0,1,2,-1,-4 };

	//nums_15 = { 0,0,0,0 };

	solution.threeSum_0(nums_15);

	/*===============    5. Longest Palindromic Substring   ===============*/
	string s_5;

	s_5 = "ccc";
	s_5 = "cbbd";
	//s_5 = "babad";
	//s_5 = "aaaaa";
	//s_5 = "aba";

	solution.longestPalindrome_0(s_5);

	/*=================     324. Wiggle Sort II 220620    =======================*/
	assert(0);

	vector<int> nums_324;

	nums_324 = { 1,5,1,1,6,4 };

	nums_324 = { 1,3,2,2,3,1 };

	//nums_324 = { 1,3,2,2,3,1,1,2,1,2,1,3 };

	//nums_324 = { 1,4,3,4,1,2,1,3,1,3,2,3,3 };

	//nums_324 = { 5,3,1,2,6,7,8,5,5 };

	solution.wiggleSort(nums_324);

	/*=================     50. Pow(x, n) 220616    =======================*/
	assert(0);
	double x_50;
	int n_50;

	x_50 = 2;
	n_50 = 10;

	x_50 = 2.1;
	n_50 = 3;

	x_50 = 2;
	n_50 = -3;

	x_50 = 0.00001;
	n_50 = 2147483647;

	x_50 = 2;
	n_50 = 2147483648;

	solution.myPow(x_50, n_50);

	/*===============     179. Largest Number 220615    ===================*/
	vector<int> nums_179;
	nums_179 = { 10,2 };
	nums_179 = { 3, 30, 34, 5, 9 };
	//nums_179 = {432, 43243};
	//nums_179 = { 0,0 };

	solution.largestNumber(nums_179);

	
	/*==============   1476. Subrectangle Queries  ================*/
	
	vector<vector<int>> rectangle_1476{{1, 2, 1}, {4, 3, 4}, {3, 2, 1}, {1, 1, 1}};
	vector<int> update_1476{ 0,0,3,2,5 };

	SubrectangleQueries* obj_1476 = new SubrectangleQueries(rectangle_1476);
	obj_1476->updateSubrectangle(update_1476[0], update_1476[1], update_1476[2], update_1476[3], update_1476[4]);

	/*============     130. Surrounded Regions   =========*/
	vector<vector<char>> board_130;

	board_130 = { {'X', 'X', 'X', 'X'}, {'X', 'O', 'O', 'X'}, {'X', 'X', 'O', 'X'}, {'X', 'O', 'X', 'X'} };

	//board_130 = { {'O', 'X', 'X', 'O', 'X'}, {'X', 'O', 'O', 'X', 'O'}, {'X', 'O', 'X', 'O', 'X'}, {'O', 'X', 'O', 'O', 'O'}, {'X', 'X', 'O', 'X', 'O'} };

	board_130 = {{'X', 'X', 'X', 'X', 'X'}, {'X', 'O', 'O', 'O', 'X'}, {'X', 'X', 'O', 'O', 'X'}, {'X', 'X', 'X', 'O', 'X'}, {'X', 'O', 'X', 'X', 'X'}};

	board_130 = {{'O', 'O', 'O', 'O', 'X', 'X'}, {'O', 'O', 'O', 'O', 'O', 'O'}, {'O', 'X', 'O', 'X', 'O', 'O'}, {'O', 'X', 'O', 'O', 'X', 'O'}, {'O', 'X', 'O', 'X', 'O', 'O'}, {'O', 'X', 'O', 'O', 'O', 'O'}};

	board_130 = {{'X', 'X', 'X', 'X', 'O', 'O', 'X', 'X', 'O'}, {'O', 'O', 'O', 'O', 'X', 'X', 'O', 'O', 'X'}, {'X', 'O', 'X', 'O', 'O', 'X', 'X', 'O', 'X'}, {'O', 'O', 'X', 'X', 'X', 'O', 'O', 'O', 'O'}, {'X', 'O', 'O', 'X', 'X', 'X', 'X', 'X', 'O'}, {'O', 'O', 'X', 'O', 'X', 'O', 'X', 'O', 'X'}, {'O', 'O', 'O', 'X', 'X', 'O', 'X', 'O', 'X'}, {'O', 'O', 'O', 'X', 'O', 'O', 'O', 'X', 'O'}, {'O', 'X', 'O', 'O', 'O', 'X', 'O', 'X', 'O'}};

	int m_130 = board_130.size();
	int n_130 = board_130[0].size();

	for (int i = 0; i < m_130; i++) {
		for (int j = 0; j < n_130; j++) {
			cout << board_130[i][j] << ", ";
		}
		cout << endl;
	}

	solution.solve(board_130);

	cout << "===================" << endl << endl;
	for (int i = 0; i < m_130; i++) {
		for (int j = 0; j < n_130; j++) {
			cout << board_130[i][j] << ", ";
		}
		cout << endl;
	}

	/*============ 152. Maximum Product Subarray =========*/
	vector<int> nums_152;

	nums_152 = { 2,3,-2,4 };
	//nums_152 = { -2,0,-1 };
	//nums_152 = { -2,3,-4 };
	//nums_152 = { 7,-2,-4 };
	//nums_152 = { -1, -2, -9, -6 };//108
	//nums_152 = {1, 2, -1, -2, 2, 1, -2, 1, 4, -5, 4}; // 1280
	nums_152 = {1, -2, 0, 1, -4, 1, 1, 5, 4, -1, 6, 4, 1, -5, 0, -1, -5, 1, -6, -4}; // 2400

	solution.maxProduct(nums_152);

	/*=================  55. Jump Game   =================*/
	vector<int> nums_55;
	nums_55 = { 3,2,1,0,4 };
	nums_55 = { 0,2,3 };
	nums_55 = { 2,0 };
	nums_55 = { 2,3,1,1,4 };
	//nums_55 = { 1,2,3 };
	nums_55 = { 0,1,1,1 };
	nums_55 = { 3,0,8,2,0,0,1 };

	//nums_55 = {6855, 5194, 16267, 13414, 19264, 3136, 16140, 5569, 3316, 7150, 12683, 12535, 15897, 6784, 14314, 12994, 19070, 19796, 11940, 14525, 1332, 376, 18596, 16866, 9196, 8458, 11632, 12426, 17737, 7823, 2332, 8604, 11580, 1873, 15238, 13145, 733, 2383, 3360, 2561, 5450, 7796, 7302, 11762, 17989, 5313, 8882, 17209, 7747, 2232, 5156, 15284, 12678, 6848, 14845, 19339, 13618, 15468, 19976, 13100, 1837, 39, 7077, 17193, 15498, 15782, 17023, 15734, 4745, 508, 6492, 8719, 5379, 1315, 12525, 9693, 19021, 5948, 17964, 6744, 19091, 19984, 14588, 18688, 12023, 9337, 15674, 1769, 6791, 16627, 9085, 8032, 10226, 1582, 186, 17181, 326, 17863, 15103, 3225, 3381, 18822, 3115, 5848, 18091, 2143, 8379, 15989, 4345, 10330, 16754, 4650, 13836, 17641, 2335, 10186, 1384, 14703, 12022, 15813, 16830, 6979, 18644, 16610, 6200, 10300, 16081, 3875, 1880, 8657, 3225, 12301, 12781, 6981, 17638, 7578, 5674, 13365, 5177, 8698, 1456, 2522, 16927, 16981, 10217, 17049, 541, 10159, 6152, 10956, 13953, 10176, 13128, 16184, 16452, 17910, 7218, 1926, 5476, 19478, 523, 6599, 14465, 19269, 8408, 17102, 2922, 18012, 11990, 14071, 7170, 438, 5320, 617, 13991, 5596, 19908, 11204, 2636, 19044, 5347, 6284, 19903, 2974, 11384, 17170, 5848, 9159, 12526, 18536, 15700, 3704, 4197, 7522, 19674, 5638, 16234, 9347, 9528, 15488, 2783, 19535, 10813, 15239, 12099, 15984, 16265, 8746, 2652, 8716, 6259, 14031, 4793, 5408, 8806, 376, 8081, 12504, 8480, 17156, 12822, 14523, 14606, 9932, 17658, 3723, 5093, 18867, 2729, 13780, 19123, 15904, 1933, 11531, 13840, 19984, 14681, 11370, 12893, 14603, 1762, 14940, 18355, 5383, 8401, 8791, 3904, 19052, 5010, 11597, 16424, 10294, 13295, 17042, 15589, 5292, 10406, 4783, 17685, 19705, 1474, 12291, 6625, 4563, 8546, 3216, 9104, 5748, 10932, 1813, 18847, 17938, 12387, 6352, 16727, 2086, 12300, 2657, 9151, 10899, 172, 173, 3587, 17700, 14470, 6943, 3672, 18525, 9235, 6336, 18412, 6176, 2536, 5248, 12833, 17035, 4211, 12802, 10653, 3410, 6788, 8463, 4493, 12819, 13463, 971, 17920, 5311, 16172, 3486, 9221, 9805, 8766, 16375, 9576, 19176, 8195, 3383, 2090, 12153, 1731, 6684, 5437, 19056, 17886, 10607, 15704, 8255, 126, 11727, 5702, 11531, 16353, 5913, 1222, 4009, 9368, 6661, 123, 8730, 3270, 19521, 3784, 19303, 15250, 5320, 13432, 7313, 10943, 16258, 1851, 19602, 18349, 14693, 3267, 10526, 19691, 13616, 5751, 4134, 7997, 16817, 5005, 7697, 10094, 17106, 1899, 2841, 16916, 7928, 5256, 15405, 17835, 14890, 16132, 12424, 3032, 18906, 2985, 13664, 5543, 8556, 13683, 13200, 14027, 11557, 13862, 16315, 6755, 11375, 13298, 1554, 11265, 12907, 11243, 13140, 11116, 12954, 11048, 3914, 9890, 7324, 4249, 9876, 3508, 6113, 17641, 14925, 8825, 11103, 12402, 11131, 14214, 19482, 4393, 5308, 5744, 13155, 7270, 11284, 21, 11835, 14712, 3454, 13436, 2405, 9368, 5555, 12542, 15524, 18210, 454, 15826, 2138, 8899, 1097, 6802, 9711, 5282, 8612, 15115, 9450, 2391, 14332, 16351, 327, 17396, 6998, 5382, 8957, 9198, 17105, 9322, 15861, 7795, 14012, 18747, 5590, 667, 10284, 9009, 10703, 16727, 15110, 19821, 18650, 13856, 1563, 7339, 11783, 4554, 2337, 10994, 1430, 4087, 5069, 9949, 3636, 5958, 6502, 13166, 12893, 1461, 10552, 18615, 18260, 12615, 5630, 11137, 17029, 10565, 17730, 13473, 10746, 6951, 4210, 2381, 4790, 10600, 9349, 876, 6876, 6370, 19251, 14081, 2124, 17997, 8969, 12785, 7050, 108, 8089, 13762, 475, 268, 6158, 13155, 13991, 15715, 5715, 8938, 10972, 14855, 6272, 13155, 12493, 17627, 8853, 15461, 2353, 16643, 16073, 17774, 6403, 1414, 7288, 19520, 3092, 688, 15303, 14257, 8071, 7477, 12278, 14362, 9252, 15793, 1733, 14522, 9306, 8751, 11917, 10644, 4582, 13836, 11449, 14919, 11631, 11015, 12770, 18207, 37, 862, 17219, 11730, 1667, 15929, 16830, 17073, 393, 15395, 18245, 19374, 13246, 3906, 15198, 19864, 7855, 17177, 19606, 18062, 5138, 4107, 13064, 19766, 6287, 10626, 15253, 11989, 12977, 5996, 3403, 8289, 8766, 5305, 13948, 18476, 11339, 6120, 11257, 5680, 6429, 18264, 8179, 9754, 1953, 9830, 674, 12375, 18518, 15049, 15704, 1563, 9472, 17195, 17596, 10121, 1894, 11838, 10296, 9935, 15159, 16269, 12224, 19475, 10664, 6162, 244, 1425, 15324, 419, 3939, 12635, 3391, 4801, 13664, 7264, 18673, 5146, 16422, 7254, 8785, 147, 15454, 12334, 6682, 12073, 18393, 1264, 12340, 8864, 6905, 11209, 14756, 11204, 1307, 11071, 16656, 4110, 3651, 7248, 3885, 12152, 6102, 8367, 13982, 9861, 4469, 15068, 913, 14235, 7987, 7106, 1121, 17485, 17841, 6419, 13805, 13079, 2567, 10073, 18703, 13607, 11235, 15063, 2706, 18040, 18762, 16531, 16759, 13798, 9853, 10009, 18668, 19829, 12508, 18827, 19404, 13209, 6514, 14022, 7748, 7506, 9108, 18547, 4620, 2344, 19681, 9714, 18168, 1692, 5972, 12013, 10737, 13567, 19711, 19072, 14128, 12566, 8278, 38, 12196, 11526, 12205, 17921, 6630, 18305, 12937, 2635, 16892, 13287, 16194, 13088, 2596, 18113, 7982, 12817, 13854, 3513, 14917, 5639, 1506, 18632, 1418, 9991, 4465, 2583, 375, 731, 7469, 11002, 16997, 7627, 4935, 9100, 7468, 6154, 5022, 11310, 19140, 18514, 15070, 13150, 9798, 14688, 17369, 2219, 1418, 13483, 9430, 5438, 15934, 19598, 12877, 10723, 6251, 14247, 11775, 1133, 11962, 14360, 17055, 1067, 14758, 8085, 15125, 16413, 17890, 18600, 17345, 6910, 9873, 8226, 16854, 17981, 16103, 17030, 17355, 17017, 8246, 2616, 7635, 8557, 10503, 7476, 7809, 3295, 14974, 18414, 4478, 17938, 12650, 16528, 1014, 8040, 3414, 7475, 1642, 19269, 4045, 12730, 12553, 18996, 16025, 3415, 8669, 14044, 11709, 1179, 9853, 12755, 13604, 15675, 2513, 8212, 8981, 16821, 14477, 8352, 15703, 1241, 1322, 2781, 17584, 15549, 5700, 19366, 19346, 5668, 6229, 6834, 4537, 1031, 7730, 3924, 11552, 16920, 8098, 3575, 1319, 13372, 18899, 15680, 10868, 4016, 1628, 12580, 12764, 11493, 18507, 10650, 12801, 4555, 17190, 4959, 15992, 2223, 7182, 17799, 12568, 9728, 13251, 6942, 18544, 16910, 4577, 4417, 901, 3999, 7749, 11525, 9973, 12807, 8150, 10150, 528, 16130, 11277, 3781, 13186, 19557, 7618, 18559, 19377, 5484, 9402, 17779, 15561, 10830, 12092, 8117, 18549, 6995, 7016, 10378, 15131, 17024, 15155, 5226, 15939, 10850, 926, 13383, 207, 8486, 13546, 501, 9295, 12810, 8533, 11436, 16061, 16436, 2636, 2581, 14129, 17659, 19841, 15813, 18968, 12294, 17239, 14344, 13666, 13857, 17690, 14274, 14766, 18584, 10854, 14818, 660, 6107, 448, 14467, 18726, 7580, 13950, 15378, 13575, 11416, 8588, 18553, 6569, 19314, 18007, 15115, 15778, 3716, 10233, 5983, 16192, 17601, 6532, 7419, 1887, 6281, 19331, 3290, 18008, 15193, 7486, 13650, 2456, 2670, 1309, 1472, 3244, 19149, 6772, 2879, 4694, 1272, 13349, 3216, 14490, 13255, 15230, 2422, 10382, 15987, 295, 4933, 11781, 17047, 364, 12380, 9101, 10647, 4014, 14615, 2865, 7621, 10193, 15524, 10436, 2466, 10005, 8719, 14635, 749, 18356, 9651, 13199, 13626, 18044, 6466, 1383, 5466, 331, 1675, 12568, 19928, 3642, 8675, 17384, 18123, 1110, 2985, 2017, 12024, 5010, 15000, 16062, 19396, 15667, 9135, 19583, 8358, 15135, 2745, 15347, 15602, 11734, 1838, 18748, 6384, 13582, 19418, 9339, 9832, 4447, 18382, 904, 10920, 2358, 8707, 18249, 15741, 12414, 10275, 14126, 8516, 4854, 14058, 14181, 10150, 1732, 1188, 4629, 17820, 19952, 4663, 14827, 5284, 10871, 7825, 13435, 9404, 5946, 10475, 2183, 8973, 15201, 16997, 14006, 8630, 12048, 19668, 9462, 16341, 19197, 1279, 10928, 12858, 1234, 1417, 4153, 10718, 3115, 15683, 15368, 11017, 14041, 16946, 19812, 3176, 15945, 16692, 19764, 6106, 641, 10175, 14157, 7311, 16582, 5993, 11519, 6136, 5490, 17508, 9635, 19178, 16680, 19282, 9528, 12313, 18728, 4752, 19800, 3699, 15185, 13856, 15372, 4646, 7653, 7702, 4227, 3614, 451, 19167, 19229, 18018, 18241, 9381, 14111, 15114, 16247, 6388, 3233, 1989, 18734, 10647, 6233, 3014, 3007, 10667, 18792, 6827, 6870, 537, 487, 14370, 8029, 8253, 13613, 10500, 6209, 16765, 4894, 4725, 12792, 15398, 12084, 305, 9496, 10879, 13445, 9194, 13883, 17452, 3230, 11978, 10841, 1800, 14214, 3979, 5733, 16041, 2691, 9122, 16940, 4847, 788, 4678, 15520, 9601, 702, 2824, 6262, 13218, 836, 2722, 14226, 7355, 12002, 8494, 8575, 16561, 7092, 487, 8060, 15035, 3129, 8153, 1200, 4214, 8181, 7066, 1795, 4698, 3293, 16754, 356, 9567, 3446, 18400, 838, 798, 9030, 17712, 18445, 4414, 13924, 12094, 16871, 15923, 6818, 8784, 18877, 244, 10653, 13804, 15928, 19825, 9731, 17516, 11260, 16837, 9793, 7685, 10616, 8998, 12015, 585, 12148, 8379, 9993, 1658, 754, 3226, 2066, 2071, 2113, 14764, 7731, 19880, 13532, 17240, 3156, 8702, 3216, 11750, 9263, 12565, 18343, 14319, 6848, 2258, 3543, 9692, 3712, 10451, 13781, 10456, 14442, 16283, 18853, 2558, 5385, 13886, 14717, 8641, 10520, 18871, 8749, 11583, 3772, 5872, 3884, 8639, 8651, 18530, 1866, 16992, 19998, 11523, 4402, 17933, 16055, 2385, 12160, 16070, 16139, 10161, 15216, 13019, 2705, 13403, 19029, 18391, 3331, 18556, 19205, 17253, 303, 1098, 12621, 15722, 10925, 3186, 18760, 17725, 7672, 16113, 16849, 15779, 1815, 6320, 3856, 4336, 4927, 3102, 11072, 17006, 4436, 13321, 13961, 2712, 19409, 10235, 15035, 14512, 8507, 11409, 3961, 18285, 14519, 11604, 13398, 13374, 13634, 10014, 14964, 10370, 8523, 7132, 17665, 18398, 10639, 4648, 19584, 9629, 9493, 16129, 6040, 19582, 9214, 10337, 4904, 11693, 15031, 14348, 19358, 2877, 19751, 9014, 6278, 3384, 19052, 12225, 5763, 3663, 7719, 11380, 9310, 7724, 1384, 4537, 6870, 14147, 5827, 19035, 15770, 19921, 763, 4049, 13362, 13313, 5522, 2784, 9680, 12013, 16384, 16268, 6853, 5854, 12757, 6983, 7503, 7791, 13900, 2039, 1427, 9550, 1575, 19524, 3132, 18492, 10051, 7404, 14847, 5430, 4532, 12973, 4676, 1546, 3672, 8580, 18472, 1075, 18372, 4357, 13260, 18857, 1998, 6640, 4178, 11517, 7247, 2787, 16790, 11575, 3670, 15266, 4659, 1658, 10678, 8479, 13638, 15278, 15470, 2428, 13555, 217, 3398, 8551, 14080, 6908, 4283, 13075, 4861, 6194, 14035, 5078, 5974, 17353, 14705, 2022, 45, 3069, 18973, 3928, 15436, 17036, 8752, 12455, 316, 18472, 10628, 1315, 12689, 9542, 18402, 16737, 11784, 11573, 7981, 19734, 16369, 1894, 9940, 11672, 14155, 7072, 9937, 10115, 2002, 13747, 5251, 12819, 18309, 1266, 1110, 5062, 15944, 2637, 2379, 13365, 209, 4803, 10359, 11948, 6165, 10668, 18131, 8269, 19286, 10565, 17086, 465, 338, 2355, 18392, 12805, 19186, 4695, 914, 15585, 17160, 4238, 8129, 7911, 9857, 14628, 1, 13992, 18722, 3189, 13298, 11617, 1983, 19754, 4074, 5174, 11946, 1144, 8169, 11039, 13022, 17931, 699, 16573, 18898, 15601, 19373, 8539, 4011, 13454, 3916, 15244, 16946, 3619, 1393, 7633, 10649, 16866, 19074, 19102, 16110, 15421, 15923, 229, 14711, 8169, 18175, 9534, 5962, 6551, 14037, 11669, 4061, 4476, 17412, 4001, 6852, 629, 4951, 5007, 11452, 2345, 172, 19877, 12418, 11144, 13481, 10447, 17846, 7046, 14650, 6039, 18708, 3659, 16655, 19050, 12561, 13879, 7385, 1015, 5200, 73, 12216, 1868, 7909, 19770, 3387, 5106, 2785, 9221, 8369, 2239, 15206, 17760, 995, 4944, 14533, 8598, 16647, 3171, 6701, 7023, 1124, 3456, 10513, 5659, 1295, 8201, 881, 3633, 3695, 12108, 1804, 5867, 5933, 12854, 2786, 17280, 56, 3852, 917, 10864, 6925, 8377, 19994, 19692, 1006, 19652, 4641, 17861, 3219, 19770, 2633, 11323, 17736, 16568, 4154, 1933, 17065, 12285, 6316, 13891, 2134, 356, 18640, 16429, 10426, 17155, 11331, 15340, 18995, 8799, 4922, 9558, 2510, 4969, 19309, 19886, 6154, 1856, 17192, 11915, 1237, 2647, 4658, 2154, 172, 16393, 8987, 19047, 5237, 13119, 19528, 5273, 3507, 11735, 4482, 12040, 9947, 19530, 4595, 4413, 7407, 1719, 19488, 15053, 7746, 14592, 12215, 13354, 3973, 315, 12940, 15979, 12890, 9644, 19891, 14012, 2950, 18230, 11845, 11139, 5240, 4499, 19739, 16756, 5530, 17807, 11518, 6249, 19050, 18639, 10852, 8818, 16883, 1299, 8718, 17207, 19882, 16240, 15039, 13315, 895, 10638, 13767, 17664, 8453, 1969, 12678, 5469, 16445, 8965, 7367, 7544, 12242, 9527, 10946, 14698, 13447, 3652, 18993, 10472, 18407, 18977, 3785, 516, 11099, 17187, 685, 11573, 11491, 3665, 4788, 16280, 17345, 18386, 5814, 11049, 16924, 6340, 13216, 2467, 621, 1332, 5921, 3614, 19716, 5462, 4893, 11339, 18079, 4293, 11027, 8854, 16374, 19322, 14336, 17033, 18633, 11910, 3139, 3659, 15468, 13685, 8379, 4363, 2437, 6307, 7987, 11611, 17646, 7184, 1008, 8775, 4278, 19677, 5325, 497, 5592, 5630, 3952, 16879, 8234, 4475, 3787, 3915, 962, 1308, 16118, 2179, 2159, 19376, 1601, 19649, 13679, 11487, 14017, 15772, 12091, 9611, 19459, 10475, 1453, 10391, 16831, 2605, 4658, 6424, 13429, 13480, 8785, 8952, 3316, 18646, 6694, 6156, 1878, 14072, 18511, 786, 1609, 5354, 8678, 1324, 2016, 1925, 8659, 10080, 9884, 14054, 15370, 7782, 17519, 10370, 7128, 10848, 4607, 536, 11237, 15685, 17993, 12707, 4287, 14870, 18063, 8561, 8045, 10967, 17212, 7944, 7777, 18810, 14945, 17566, 7497, 13650, 10209, 12069, 5474, 10687, 16834, 11532, 18793, 6341, 4637, 15450, 3257, 5694, 15061, 7558, 13926, 14406, 14578, 19986, 1917, 16722, 8274, 7043, 19793, 11693, 8072, 19303, 19916, 4059, 11709, 3063, 7853, 18240, 8285, 17571, 14704, 17274, 3105, 5984, 3151, 7113, 176, 1561, 3235, 8346, 14035, 4667, 16437, 13857, 1190, 19005, 9849, 10124, 3473, 15760, 14933, 10633, 8414, 7286, 12621, 15813, 8313, 2185, 13938, 2665, 16356, 16054, 6944, 10643, 17237, 13709, 3617, 3050, 9786, 523, 12534, 9409, 10814, 9150, 11011, 4045, 4595, 17885, 10603, 18952, 531, 16121, 12380, 7672, 8247, 5366, 13561, 15561, 8150, 9038, 5810, 16327, 3489, 14896, 15504, 2031, 12537, 15761, 6865, 5161, 7767, 6087, 4305, 2804, 15524, 13853, 4831, 19493, 15153, 18770, 16753, 7734, 17160, 6271, 5855, 7730, 11442, 13654, 19714, 7110, 11548, 14492, 8256, 867, 4913, 9050, 9974, 9557, 15854, 1143, 19324, 9216, 19072, 4402, 18709, 15361, 427, 2531, 4586, 18552, 6231, 16617, 11160, 10520, 13145, 9003, 579, 19478, 1200, 64, 3245, 17818, 3177, 5536, 4582, 14032, 12821, 9318, 4018, 16702, 3670, 912, 745, 9382, 17435, 5211, 18093, 12729, 15881, 14867, 12433, 11718, 4999, 9202, 9648, 7147, 19508, 15198, 15586, 10264, 18225, 3840, 18430, 1563, 15981, 11852, 12805, 15648, 9353, 17906, 2001, 2639, 583, 19773, 19816, 12556, 2117, 9531, 2597, 13215, 3048, 15178, 11405, 994, 8696, 8630, 19579, 9368, 18057, 13753, 19847, 12714, 12968, 1959, 10626, 15880, 16242, 18448, 2222, 8812, 211, 4012, 13976, 17218, 17771, 13636, 11693, 15033, 1177, 12578, 8390, 4331, 12707, 17596, 2987, 15122, 6594, 4192, 3029, 11554, 18449, 2751, 4368, 17715, 2010, 5629, 13055, 9483, 19640, 11028, 5630, 2186, 4954, 3500, 2650, 16161, 11933, 12559, 15143, 6555, 4380, 16455, 11751, 14172, 9354, 9230, 911, 1325, 4512, 17477, 13464, 3421, 16614, 16545, 2370, 13886, 12332, 35, 5574, 7874, 5054, 6662, 15189, 10848, 15355, 12971, 14775, 10450, 5014, 16795, 4754, 18437, 14064, 4401, 13366, 1237, 12333, 1596, 18346, 17508, 18888, 9152, 2936, 7920, 14575, 1307, 11674, 14168, 1028, 9526, 14594, 2879, 16419, 8709, 1084, 14029, 2430, 15457, 14159, 5506, 38, 6241, 6909, 10955, 14387, 1642, 17545, 19157, 15964, 17135, 139, 17635, 19051, 18501, 8707, 15026, 555, 4346, 7910, 3132, 5157, 17505, 3131, 12273, 10885, 16684, 15225, 12031, 8102, 11221, 4121, 14417, 19379, 683, 7066, 8516, 8071, 10451, 18555, 17567, 18598, 19273, 15851, 10745, 19765, 3300, 8484, 8694, 4149, 8377, 346, 7239, 13941, 7516, 11232, 12440, 2909, 3333, 3407, 12754, 9616, 9105, 9195, 14636, 1768, 11985, 11224, 7768, 15747, 15092, 9634, 1173, 15175, 17508, 19325, 12580, 15823, 16691, 13273, 13072, 15013, 18681, 12652, 17768, 19188, 8548, 19470, 10311, 7885, 14977, 9394, 1220, 17450, 18217, 3856, 17266, 9409, 7496, 2333, 13072, 7319, 2582, 4626, 11986, 19923, 16414, 49, 15874, 6376, 17841, 17287, 2648, 13204, 17798, 4122, 4396, 7239, 17108, 14981, 3901, 4552, 10860, 10312, 4254, 18155, 570, 16291, 5547, 4438, 10684, 8668, 6887, 16721, 7810, 7109, 8590, 18981, 1113, 17788, 8355, 46, 2886, 10379, 322, 13878, 13835, 7254, 15893, 9337, 18482, 10656, 3225, 5381, 9819, 8688, 18454, 5378, 2956, 18093, 16225, 6412, 17694, 4090, 11728, 11107, 10290, 15717, 14691, 10108, 7508, 10868, 9381, 8898, 7470, 10357, 9683, 880, 18203, 11442, 5820, 2391, 10434, 
		//11314, 4887, 16956, 10876, 1544, 7513, 1213, 100, 232, 3576};

	solution.canJump(nums_55);

	/*======  33. Search in Rotated Sorted Array  ======*/
	vector<int> nums_33 = {4, 5, 6, 7, 0, 1, 2};
	int target_33 = 0;

	nums_33 = {1};
	target_33 = 1;

	nums_33 = {3, 1};
	target_33 = 1;

	/*nums_33 = { 3, 5, 1 };
	target_33 = 3;

	nums_33 = { 4,5,6,7,8,1,2,3 };
	target_33 = 8;*/

	solution.search(nums_33, target_33);

	/*======  19. Remove Nth Node From End of List  ======*/
	int n_19 = 1;
	n_19 = 1;
	//n_19 = 2;

	//vector<Solution::ListNode> l_19_vector = { 1,2,3,4,5 };
	vector<Solution::ListNode> l_19_vector = { 1};
	//vector<Solution::ListNode> l_19_vector = { 1,2 };

	Solution::ListNode* head_19 = &l_19_vector[0];

	if (l_19_vector.size() > 1) {
		Solution::ListNode* p_19 = head_19;
		for (auto&& au : l_19_vector) {
			p_19->next = &au;
			p_19 = p_19->next;
		}
	}
	

	solution.removeNthFromEnd(head_19, n_19);

	/*===================  2. Add Two Numbers  ==========================*/
	
	vector<Solution::ListNode> l_2_1 = { 2,4,3 };
	vector<Solution::ListNode> l_2_2 = { 5,6,4 };
	Solution::ListNode* head_1 = &l_2_1[0];
	Solution::ListNode* head_2 = &l_2_2[0];
	Solution::ListNode* p_1 = head_1;
	Solution::ListNode* p_2 = head_2;

	for (auto&& au : l_2_1) {
		p_1->next = &au;
		p_1 = p_1->next;
	}

	for (auto&& au : l_2_2) {
		p_2->next = &au;
		p_2 = p_2->next;
	}

	solution.addTwoNumbers(head_1, head_2);

	/*===================  79. Word Search  ==========================*/

	vector<vector<char>> board_79 = { {'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'} };
	string word_79 = "ABCCED";

	board_79 = { {'A', 'B', 'C', 'E'}, {'S', 'F', 'C', 'S'}, {'A', 'D', 'E', 'E'} };
	word_79 = "ABCB";

	solution.exist(board_79, word_79);

	/*============== 146. LRU Cache 22061 ================*/
	int capacity_146 = 2;
	vector<vector<int>> put_146{ { 1,1 }, { 2,2 }, { 3,3 }, { 4,4 }};
	vector<int> get_146{ 1,2,1,3,4 };

	LRUCache* obj = new LRUCache(capacity_146);
	
	obj->put(put_146[0][0], put_146[0][1]);
	obj->put(put_146[1][0], put_146[1][1]);
	int param_1 = obj->get(get_146[0]);
	obj->put(put_146[2][0], put_146[2][1]);
	param_1 = obj->get(get_146[1]);
	obj->put(put_146[3][0], put_146[3][1]);
	param_1 = obj->get(get_146[2]);
	param_1 = obj->get(get_146[3]);
	param_1 = obj->get(get_146[4]);
	obj->put(3,6);

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
	k = 3;
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

	
	target = 0;
	bool modified = false;
	s = "()";
	

	strs = { "flower","flow","flight" };
	cout << strs.size() << " "<< strs[1].size() << endl;

	
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
