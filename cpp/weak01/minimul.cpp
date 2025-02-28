#include <iostream>

using namespace std;

int mul(int a, int b) {
    return a * b;
}

int main(){
    int a, b;
    int result;

    cout << "Enter two numbers: ";
    cin >> a >> b;

    result = mul(a, b);

    cout << "The result is: " << result << endl;
    return 0;
}