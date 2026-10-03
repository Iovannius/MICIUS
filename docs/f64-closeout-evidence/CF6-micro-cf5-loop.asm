140099110: 44 0f b6 11                 	movzx	r10d, byte ptr [rcx]
140099114: 41 c1 e2 05                 	shl	r10d, 0x5
140099118: c4 41 7d 6f 14 02           	vmovdqa	ymm10, ymmword ptr [r10 + rax]
14009911e: 44 0f b6 51 01              	movzx	r10d, byte ptr [rcx + 0x1]
140099123: 41 c1 e2 05                 	shl	r10d, 0x5
140099127: c4 41 7d 6f 0c 02           	vmovdqa	ymm9, ymmword ptr [r10 + rax]
14009912d: c4 e2 7d 5a 41 02           	vbroadcasti128	ymm0, xmmword ptr [rcx + 0x2] # ymm0 = mem[0,1,0,1]
140099133: c4 c2 7d 00 c5              	vpshufb	ymm0, ymm0, ymm13
140099138: c4 41 20 57 db              	vxorps	xmm11, xmm11, xmm11
14009913d: c4 42 7d 36 da              	vpermd	ymm11, ymm0, ymm10
140099142: c5 25 fe e0                 	vpaddd	ymm12, ymm11, ymm0
140099146: c5 fd 10 82 20 ff ff ff     	vmovupd	ymm0, ymmword ptr [rdx - 0xe0]
14009914e: c5 7d 10 9a 40 ff ff ff     	vmovupd	ymm11, ymmword ptr [rdx - 0xc0]
140099156: c4 62 fd a8 e1              	vfmadd213pd	ymm12, ymm0, ymm1 # ymm12 = (ymm0 * ymm12) + ymm1
14009915b: c5 7d 29 24 24              	vmovapd	ymmword ptr [rsp], ymm12
140099160: c4 e2 7d 5a 49 0e           	vbroadcasti128	ymm1, xmmword ptr [rcx + 0xe] # ymm1 = mem[0,1,0,1]
140099166: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
14009916b: c4 41 18 57 e4              	vxorps	xmm12, xmm12, xmm12
140099170: c4 42 75 36 e2              	vpermd	ymm12, ymm1, ymm10
140099175: c5 1d fe f1                 	vpaddd	ymm14, ymm12, ymm1
140099179: c4 62 a5 a8 f2              	vfmadd213pd	ymm14, ymm11, ymm2 # ymm14 = (ymm11 * ymm14) + ymm2
14009917e: c4 e2 7d 5a 49 1a           	vbroadcasti128	ymm1, xmmword ptr [rcx + 0x1a] # ymm1 = mem[0,1,0,1]
140099184: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
140099189: c5 e8 57 d2                 	vxorps	xmm2, xmm2, xmm2
14009918d: c4 c2 75 36 d1              	vpermd	ymm2, ymm1, ymm9
140099192: c5 6d fe f9                 	vpaddd	ymm15, ymm2, ymm1
140099196: c5 fd 10 8a 60 ff ff ff     	vmovupd	ymm1, ymmword ptr [rdx - 0xa0]
14009919e: c4 62 fd a8 fd              	vfmadd213pd	ymm15, ymm0, ymm5 # ymm15 = (ymm0 * ymm15) + ymm5
1400991a3: c4 e2 7d 5a 41 26           	vbroadcasti128	ymm0, xmmword ptr [rcx + 0x26] # ymm0 = mem[0,1,0,1]
1400991a9: c4 c2 7d 00 c5              	vpshufb	ymm0, ymm0, ymm13
1400991ae: c5 e8 57 d2                 	vxorps	xmm2, xmm2, xmm2
1400991b2: c4 c2 7d 36 d1              	vpermd	ymm2, ymm0, ymm9
1400991b7: c5 ed fe c0                 	vpaddd	ymm0, ymm2, ymm0
1400991bb: c5 fd 10 52 80              	vmovupd	ymm2, ymmword ptr [rdx - 0x80]
1400991c0: c4 e2 a5 a8 c4              	vfmadd213pd	ymm0, ymm11, ymm4 # ymm0 = (ymm11 * ymm0) + ymm4
1400991c5: c4 e2 7d 5a 61 32           	vbroadcasti128	ymm4, xmmword ptr [rcx + 0x32] # ymm4 = mem[0,1,0,1]
1400991cb: c4 c2 5d 00 e5              	vpshufb	ymm4, ymm4, ymm13
1400991d0: c5 d0 57 ed                 	vxorps	xmm5, xmm5, xmm5
1400991d4: c4 c2 5d 36 ea              	vpermd	ymm5, ymm4, ymm10
1400991d9: c5 55 fe dc                 	vpaddd	ymm11, ymm5, ymm4
1400991dd: c4 62 f5 a8 db              	vfmadd213pd	ymm11, ymm1, ymm3 # ymm11 = (ymm1 * ymm11) + ymm3
1400991e2: c4 e2 7d 5a 59 3e           	vbroadcasti128	ymm3, xmmword ptr [rcx + 0x3e] # ymm3 = mem[0,1,0,1]
1400991e8: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400991ed: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
1400991f1: c4 c2 65 36 e2              	vpermd	ymm4, ymm3, ymm10
1400991f6: c5 dd fe db                 	vpaddd	ymm3, ymm4, ymm3
1400991fa: c4 c2 ed a8 d8              	vfmadd213pd	ymm3, ymm2, ymm8 # ymm3 = (ymm2 * ymm3) + ymm8
1400991ff: c5 fd 29 5c 24 20           	vmovapd	ymmword ptr [rsp + 0x20], ymm3
140099205: c4 e2 7d 5a 59 4a           	vbroadcasti128	ymm3, xmmword ptr [rcx + 0x4a] # ymm3 = mem[0,1,0,1]
14009920b: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
140099210: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
140099214: c4 c2 65 36 e1              	vpermd	ymm4, ymm3, ymm9
140099219: c5 5d fe c3                 	vpaddd	ymm8, ymm4, ymm3
14009921d: c4 62 f5 a8 c6              	vfmadd213pd	ymm8, ymm1, ymm6 # ymm8 = (ymm1 * ymm8) + ymm6
140099222: c4 e2 7d 5a 49 56           	vbroadcasti128	ymm1, xmmword ptr [rcx + 0x56] # ymm1 = mem[0,1,0,1]
140099228: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
14009922d: c5 e0 57 db                 	vxorps	xmm3, xmm3, xmm3
140099231: c4 c2 75 36 d9              	vpermd	ymm3, ymm1, ymm9
140099236: c5 65 fe e1                 	vpaddd	ymm12, ymm3, ymm1
14009923a: c4 62 ed a8 e7              	vfmadd213pd	ymm12, ymm2, ymm7 # ymm12 = (ymm2 * ymm12) + ymm7
14009923f: c5 fd 10 5a a0              	vmovupd	ymm3, ymmword ptr [rdx - 0x60]
140099244: c5 fd 10 72 c0              	vmovupd	ymm6, ymmword ptr [rdx - 0x40]
140099249: c4 e2 7d 5a 49 62           	vbroadcasti128	ymm1, xmmword ptr [rcx + 0x62] # ymm1 = mem[0,1,0,1]
14009924f: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
140099254: c5 e8 57 d2                 	vxorps	xmm2, xmm2, xmm2
140099258: c4 c2 75 36 d2              	vpermd	ymm2, ymm1, ymm10
14009925d: c5 ed fe c9                 	vpaddd	ymm1, ymm2, ymm1
140099261: c4 e2 e5 a8 0c 24           	vfmadd213pd	ymm1, ymm3, ymmword ptr [rsp] # ymm1 = (ymm3 * ymm1) + mem
140099267: c4 e2 7d 5a 51 6e           	vbroadcasti128	ymm2, xmmword ptr [rcx + 0x6e] # ymm2 = mem[0,1,0,1]
14009926d: c4 c2 6d 00 d5              	vpshufb	ymm2, ymm2, ymm13
140099272: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
140099276: c4 c2 6d 36 e2              	vpermd	ymm4, ymm2, ymm10
14009927b: c5 dd fe d2                 	vpaddd	ymm2, ymm4, ymm2
14009927f: c4 c2 cd a8 d6              	vfmadd213pd	ymm2, ymm6, ymm14 # ymm2 = (ymm6 * ymm2) + ymm14
140099284: c4 e2 7d 5a 61 7a           	vbroadcasti128	ymm4, xmmword ptr [rcx + 0x7a] # ymm4 = mem[0,1,0,1]
14009928a: c4 c2 5d 00 e5              	vpshufb	ymm4, ymm4, ymm13
14009928f: c5 d0 57 ed                 	vxorps	xmm5, xmm5, xmm5
140099293: c4 c2 5d 36 e9              	vpermd	ymm5, ymm4, ymm9
140099298: c5 d5 fe ec                 	vpaddd	ymm5, ymm5, ymm4
14009929c: c4 c2 e5 a8 ef              	vfmadd213pd	ymm5, ymm3, ymm15 # ymm5 = (ymm3 * ymm5) + ymm15
1400992a1: c4 e2 7d 5a 99 86 00 00 00  	vbroadcasti128	ymm3, xmmword ptr [rcx + 0x86] # ymm3 = mem[0,1,0,1]
1400992aa: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400992af: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
1400992b3: c4 c2 65 36 e1              	vpermd	ymm4, ymm3, ymm9
1400992b8: c5 dd fe e3                 	vpaddd	ymm4, ymm4, ymm3
1400992bc: c4 e2 cd a8 e0              	vfmadd213pd	ymm4, ymm6, ymm0 # ymm4 = (ymm6 * ymm4) + ymm0
1400992c1: c5 fd 10 42 e0              	vmovupd	ymm0, ymmword ptr [rdx - 0x20]
1400992c6: c4 e2 7d 5a 99 92 00 00 00  	vbroadcasti128	ymm3, xmmword ptr [rcx + 0x92] # ymm3 = mem[0,1,0,1]
1400992cf: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400992d4: c5 c8 57 f6                 	vxorps	xmm6, xmm6, xmm6
1400992d8: c4 c2 65 36 f2              	vpermd	ymm6, ymm3, ymm10
1400992dd: c5 cd fe db                 	vpaddd	ymm3, ymm6, ymm3
1400992e1: c4 c2 fd a8 db              	vfmadd213pd	ymm3, ymm0, ymm11 # ymm3 = (ymm0 * ymm3) + ymm11
1400992e6: c4 e2 7d 5a b1 aa 00 00 00  	vbroadcasti128	ymm6, xmmword ptr [rcx + 0xaa] # ymm6 = mem[0,1,0,1]
1400992ef: c4 c2 4d 00 f5              	vpshufb	ymm6, ymm6, ymm13
1400992f4: c4 c2 4d 36 f9              	vpermd	ymm7, ymm6, ymm9
1400992f9: c5 c5 fe f6                 	vpaddd	ymm6, ymm7, ymm6
1400992fd: c4 c2 fd a8 f0              	vfmadd213pd	ymm6, ymm0, ymm8 # ymm6 = (ymm0 * ymm6) + ymm8
140099302: c4 e2 7d 5a 81 9e 00 00 00  	vbroadcasti128	ymm0, xmmword ptr [rcx + 0x9e] # ymm0 = mem[0,1,0,1]
14009930b: c4 c2 7d 00 c5              	vpshufb	ymm0, ymm0, ymm13
140099310: c5 c0 57 ff                 	vxorps	xmm7, xmm7, xmm7
140099314: c4 c2 7d 36 fa              	vpermd	ymm7, ymm0, ymm10
140099319: c5 45 fe c0                 	vpaddd	ymm8, ymm7, ymm0
14009931d: c5 fd 10 02                 	vmovupd	ymm0, ymmword ptr [rdx]
140099321: c4 62 fd a8 44 24 20        	vfmadd213pd	ymm8, ymm0, ymmword ptr [rsp + 0x20] # ymm8 = (ymm0 * ymm8) + mem
140099328: c4 e2 7d 5a b9 b6 00 00 00  	vbroadcasti128	ymm7, xmmword ptr [rcx + 0xb6] # ymm7 = mem[0,1,0,1]
140099331: c4 c2 45 00 fd              	vpshufb	ymm7, ymm7, ymm13
140099336: c4 42 45 36 c9              	vpermd	ymm9, ymm7, ymm9
14009933b: c5 b5 fe ff                 	vpaddd	ymm7, ymm9, ymm7
14009933f: c4 c2 fd a8 fc              	vfmadd213pd	ymm7, ymm0, ymm12 # ymm7 = (ymm0 * ymm7) + ymm12
140099344: 48 81 c2 00 01 00 00        	add	rdx, 0x100
14009934b: 48 81 c1 c2 00 00 00        	add	rcx, 0xc2
140099352: 49 ff c8                    	dec	r8
140099355: 0f 85 b5 fd ff ff           	jne	0x140099110 <cf5_pair_kernel+0x90>
