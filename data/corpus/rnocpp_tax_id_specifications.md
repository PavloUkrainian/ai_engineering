# Ukrainian Tax ID (RNOCPP): Algorithmic Structure and Checksum Verification

> **Document Type:** technical_guide  
> **Effective Date / Version:** 2021-08-25  

---

## 1. Ten-Digit RNOCPP Structural Composition

The Ukrainian Registration Number of the Taxpayer's Record Card (РНОКПП) comprises exactly 10 decimal digits. The first 5 digits represent the sequential number of days elapsed from December 31, 1899 to the individual's date of birth. Digits 6-9 represent the sequential record number in the register for persons born on that day (digit 9 designates gender: odd for male, even for female). Digit 10 is the algorithmic check digit.

## 2. Checksum Mathematical Formula

The 10th check digit is computed using constant weight coefficients [-1, 5, 7, 9, 4, 6, 10, 5, 7]. Sum S = sum(d[i] * w[i]) for i=0..8. Remainder R = S % 11. If R < 10, check digit = R. If R == 10, a secondary calculation with weights [3, 5, 7, 9, 4, 6, 10, 5, 7] is evaluated: S2 = sum(d[i] * w2[i]), R2 = S2 % 11, check digit = R2 % 10.
