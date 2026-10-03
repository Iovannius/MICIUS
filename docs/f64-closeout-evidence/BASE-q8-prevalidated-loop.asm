1400d1e90: c4 a1 7a 10 34 2a           	vmovss	xmm6, dword ptr [rdx + r13]
1400d1e96: c5 ca 5a f6                 	vcvtss2sd	xmm6, xmm6, xmm6
1400d1e9a: c4 e2 7d 19 f6              	vbroadcastsd	ymm6, xmm6
1400d1e9f: c4 a1 7a 10 3c 2b           	vmovss	xmm7, dword ptr [rbx + r13]
1400d1ea5: c5 c2 5a ff                 	vcvtss2sd	xmm7, xmm7, xmm7
1400d1ea9: c4 22 79 21 64 69 e4        	vpmovsxbd	xmm12, dword ptr [rcx + 2*r13 - 0x1c]
1400d1eb0: c4 41 7e e6 ec              	vcvtdq2pd	ymm13, xmm12
1400d1eb5: c4 e2 7d 19 ff              	vbroadcastsd	ymm7, xmm7
1400d1eba: c5 7d 10 b0 20 ff ff ff     	vmovupd	ymm14, ymmword ptr [rax - 0xe0]
1400d1ec2: c5 7d 10 b8 40 ff ff ff     	vmovupd	ymm15, ymmword ptr [rax - 0xc0]
1400d1eca: c5 fd 10 80 60 ff ff ff     	vmovupd	ymm0, ymmword ptr [rax - 0xa0]
1400d1ed2: c5 7d 10 60 80              	vmovupd	ymm12, ymmword ptr [rax - 0x80]
1400d1ed7: c4 41 0d 59 ed              	vmulpd	ymm13, ymm14, ymm13
1400d1edc: c5 15 59 ee                 	vmulpd	ymm13, ymm13, ymm6
1400d1ee0: c4 41 25 58 dd              	vaddpd	ymm11, ymm11, ymm13
1400d1ee5: c4 22 79 21 6c 6e e4        	vpmovsxbd	xmm13, dword ptr [rsi + 2*r13 - 0x1c]
1400d1eec: c4 41 7e e6 ed              	vcvtdq2pd	ymm13, xmm13
1400d1ef1: c4 41 0d 59 ed              	vmulpd	ymm13, ymm14, ymm13
1400d1ef6: c5 15 59 ef                 	vmulpd	ymm13, ymm13, ymm7
1400d1efa: c4 22 79 21 74 69 e8        	vpmovsxbd	xmm14, dword ptr [rcx + 2*r13 - 0x18]
1400d1f01: c4 41 7e e6 f6              	vcvtdq2pd	ymm14, xmm14
1400d1f06: c5 95 58 e4                 	vaddpd	ymm4, ymm13, ymm4
1400d1f0a: c4 41 05 59 ee              	vmulpd	ymm13, ymm15, ymm14
1400d1f0f: c5 15 59 ee                 	vmulpd	ymm13, ymm13, ymm6
1400d1f13: c4 41 2d 58 d5              	vaddpd	ymm10, ymm10, ymm13
1400d1f18: c4 22 79 21 6c 6e e8        	vpmovsxbd	xmm13, dword ptr [rsi + 2*r13 - 0x18]
1400d1f1f: c4 41 7e e6 ed              	vcvtdq2pd	ymm13, xmm13
1400d1f24: c4 41 05 59 ed              	vmulpd	ymm13, ymm15, ymm13
1400d1f29: c5 15 59 ef                 	vmulpd	ymm13, ymm13, ymm7
1400d1f2d: c4 22 79 21 74 69 ec        	vpmovsxbd	xmm14, dword ptr [rcx + 2*r13 - 0x14]
1400d1f34: c5 95 58 db                 	vaddpd	ymm3, ymm13, ymm3
1400d1f38: c4 41 7e e6 ee              	vcvtdq2pd	ymm13, xmm14
1400d1f3d: c5 15 59 e8                 	vmulpd	ymm13, ymm13, ymm0
1400d1f41: c5 15 59 ee                 	vmulpd	ymm13, ymm13, ymm6
1400d1f45: c4 41 35 58 cd              	vaddpd	ymm9, ymm9, ymm13
1400d1f4a: c4 22 79 21 6c 6e ec        	vpmovsxbd	xmm13, dword ptr [rsi + 2*r13 - 0x14]
1400d1f51: c4 41 7e e6 ed              	vcvtdq2pd	ymm13, xmm13
1400d1f56: c5 95 59 c0                 	vmulpd	ymm0, ymm13, ymm0
1400d1f5a: c5 c5 59 c0                 	vmulpd	ymm0, ymm7, ymm0
1400d1f5e: c5 d5 58 e8                 	vaddpd	ymm5, ymm5, ymm0
1400d1f62: c4 a2 79 21 44 69 f0        	vpmovsxbd	xmm0, dword ptr [rcx + 2*r13 - 0x10]
1400d1f69: c5 fe e6 c0                 	vcvtdq2pd	ymm0, xmm0
1400d1f6d: c5 9d 59 c0                 	vmulpd	ymm0, ymm12, ymm0
1400d1f71: c5 cd 59 c0                 	vmulpd	ymm0, ymm6, ymm0
1400d1f75: c5 3d 58 c0                 	vaddpd	ymm8, ymm8, ymm0
1400d1f79: c4 a2 79 21 44 6e f0        	vpmovsxbd	xmm0, dword ptr [rsi + 2*r13 - 0x10]
1400d1f80: c5 fe e6 c0                 	vcvtdq2pd	ymm0, xmm0
1400d1f84: c5 9d 59 c0                 	vmulpd	ymm0, ymm12, ymm0
1400d1f88: c5 c5 59 c0                 	vmulpd	ymm0, ymm7, ymm0
1400d1f8c: c5 ed 58 d0                 	vaddpd	ymm2, ymm2, ymm0
1400d1f90: c5 fd 10 40 a0              	vmovupd	ymm0, ymmword ptr [rax - 0x60]
1400d1f95: c4 22 79 21 64 69 f4        	vpmovsxbd	xmm12, dword ptr [rcx + 2*r13 - 0xc]
1400d1f9c: c4 41 7e e6 e4              	vcvtdq2pd	ymm12, xmm12
1400d1fa1: c5 1d 59 e0                 	vmulpd	ymm12, ymm12, ymm0
1400d1fa5: c5 1d 59 e6                 	vmulpd	ymm12, ymm12, ymm6
1400d1fa9: c4 22 79 21 6c 6e f4        	vpmovsxbd	xmm13, dword ptr [rsi + 2*r13 - 0xc]
1400d1fb0: c4 41 25 58 dc              	vaddpd	ymm11, ymm11, ymm12
1400d1fb5: c4 41 7e e6 e5              	vcvtdq2pd	ymm12, xmm13
1400d1fba: c5 9d 59 c0                 	vmulpd	ymm0, ymm12, ymm0
1400d1fbe: c5 c5 59 c0                 	vmulpd	ymm0, ymm7, ymm0
1400d1fc2: c5 dd 58 e0                 	vaddpd	ymm4, ymm4, ymm0
1400d1fc6: c5 fd 10 40 c0              	vmovupd	ymm0, ymmword ptr [rax - 0x40]
1400d1fcb: c4 22 79 21 64 69 f8        	vpmovsxbd	xmm12, dword ptr [rcx + 2*r13 - 0x8]
1400d1fd2: c4 41 7e e6 e4              	vcvtdq2pd	ymm12, xmm12
1400d1fd7: c5 1d 59 e0                 	vmulpd	ymm12, ymm12, ymm0
1400d1fdb: c5 1d 59 e6                 	vmulpd	ymm12, ymm12, ymm6
1400d1fdf: c4 41 2d 58 d4              	vaddpd	ymm10, ymm10, ymm12
1400d1fe4: c4 22 79 21 64 6e f8        	vpmovsxbd	xmm12, dword ptr [rsi + 2*r13 - 0x8]
1400d1feb: c4 41 7e e6 e4              	vcvtdq2pd	ymm12, xmm12
1400d1ff0: c5 9d 59 c0                 	vmulpd	ymm0, ymm12, ymm0
1400d1ff4: c5 c5 59 c0                 	vmulpd	ymm0, ymm7, ymm0
1400d1ff8: c5 e5 58 d8                 	vaddpd	ymm3, ymm3, ymm0
1400d1ffc: c5 fd 10 40 e0              	vmovupd	ymm0, ymmword ptr [rax - 0x20]
1400d2001: c4 22 79 21 64 69 fc        	vpmovsxbd	xmm12, dword ptr [rcx + 2*r13 - 0x4]
1400d2008: c4 41 7e e6 e4              	vcvtdq2pd	ymm12, xmm12
1400d200d: c5 1d 59 e0                 	vmulpd	ymm12, ymm12, ymm0
1400d2011: c5 1d 59 e6                 	vmulpd	ymm12, ymm12, ymm6
1400d2015: c4 22 79 21 6c 6e fc        	vpmovsxbd	xmm13, dword ptr [rsi + 2*r13 - 0x4]
1400d201c: c4 41 35 58 cc              	vaddpd	ymm9, ymm9, ymm12
1400d2021: c4 41 7e e6 e5              	vcvtdq2pd	ymm12, xmm13
1400d2026: c5 9d 59 c0                 	vmulpd	ymm0, ymm12, ymm0
1400d202a: c5 c5 59 c0                 	vmulpd	ymm0, ymm7, ymm0
1400d202e: c5 d5 58 e8                 	vaddpd	ymm5, ymm5, ymm0
1400d2032: c5 fd 10 00                 	vmovupd	ymm0, ymmword ptr [rax]
1400d2036: c4 22 79 21 24 69           	vpmovsxbd	xmm12, dword ptr [rcx + 2*r13]
1400d203c: c4 41 7e e6 e4              	vcvtdq2pd	ymm12, xmm12
1400d2041: c5 1d 59 e0                 	vmulpd	ymm12, ymm12, ymm0
1400d2045: c5 9d 59 f6                 	vmulpd	ymm6, ymm12, ymm6
1400d2049: c5 3d 58 c6                 	vaddpd	ymm8, ymm8, ymm6
1400d204d: c4 a2 79 21 34 6e           	vpmovsxbd	xmm6, dword ptr [rsi + 2*r13]
1400d2053: c5 fe e6 f6                 	vcvtdq2pd	ymm6, xmm6
1400d2057: c5 fd 59 c6                 	vmulpd	ymm0, ymm0, ymm6
1400d205b: c5 c5 59 c0                 	vmulpd	ymm0, ymm7, ymm0
1400d205f: c5 ed 58 d0                 	vaddpd	ymm2, ymm2, ymm0
1400d2063: 49 83 c5 10                 	add	r13, 0x10
1400d2067: 48 05 00 01 00 00           	add	rax, 0x100
1400d206d: 4d 39 ef                    	cmp	r15, r13
1400d2070: 0f 85 1a fe ff ff           	jne	0x1400d1e90 <.text+0xd0e90>
