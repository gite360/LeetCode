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
	cout << "1 current id: " << 55555555555 << ", vector size: " <<555555555555 << endl;


#endif


	Solution solution;

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
