1400d3ee0: 42 0f b6 34 11              	movzx	esi, byte ptr [rcx + r10]
1400d3ee5: c1 e6 05                    	shl	esi, 0x5
1400d3ee8: c4 a1 7d 6f 34 1e           	vmovdqa	ymm6, ymmword ptr [rsi + r11]
1400d3eee: 42 0f b6 74 11 01           	movzx	esi, byte ptr [rcx + r10 + 0x1]
1400d3ef4: c1 e6 05                    	shl	esi, 0x5
1400d3ef7: c4 a1 7d 6f 2c 1e           	vmovdqa	ymm5, ymmword ptr [rsi + r11]
1400d3efd: c4 a2 7d 5a 44 11 02        	vbroadcasti128	ymm0, xmmword ptr [rcx + r10 + 0x2] # ymm0 = mem[0,1,0,1]
1400d3f04: c4 c2 7d 00 c5              	vpshufb	ymm0, ymm0, ymm13
1400d3f09: c4 41 20 57 db              	vxorps	xmm11, xmm11, xmm11
1400d3f0e: c4 62 7d 36 de              	vpermd	ymm11, ymm0, ymm6
1400d3f13: c5 25 fe e0                 	vpaddd	ymm12, ymm11, ymm0
1400d3f17: c5 fd 10 82 20 ff ff ff     	vmovupd	ymm0, ymmword ptr [rdx - 0xe0]
1400d3f1f: c5 7d 10 9a 40 ff ff ff     	vmovupd	ymm11, ymmword ptr [rdx - 0xc0]
1400d3f27: c4 42 fd a8 e1              	vfmadd213pd	ymm12, ymm0, ymm9 # ymm12 = (ymm0 * ymm12) + ymm9
1400d3f2c: c5 7d 29 24 24              	vmovapd	ymmword ptr [rsp], ymm12
1400d3f31: c4 22 7d 5a 4c 11 0e        	vbroadcasti128	ymm9, xmmword ptr [rcx + r10 + 0xe] # ymm9 = mem[0,1,0,1]
1400d3f38: c4 42 35 00 cd              	vpshufb	ymm9, ymm9, ymm13
1400d3f3d: c4 41 18 57 e4              	vxorps	xmm12, xmm12, xmm12
1400d3f42: c4 62 35 36 e6              	vpermd	ymm12, ymm9, ymm6
1400d3f47: c4 41 1d fe f1              	vpaddd	ymm14, ymm12, ymm9
1400d3f4c: c4 42 a5 a8 f0              	vfmadd213pd	ymm14, ymm11, ymm8 # ymm14 = (ymm11 * ymm14) + ymm8
1400d3f51: c4 22 7d 5a 44 11 1a        	vbroadcasti128	ymm8, xmmword ptr [rcx + r10 + 0x1a] # ymm8 = mem[0,1,0,1]
1400d3f58: c4 42 3d 00 c5              	vpshufb	ymm8, ymm8, ymm13
1400d3f5d: c4 41 30 57 c9              	vxorps	xmm9, xmm9, xmm9
1400d3f62: c4 62 3d 36 cd              	vpermd	ymm9, ymm8, ymm5
1400d3f67: c4 41 35 fe f8              	vpaddd	ymm15, ymm9, ymm8
1400d3f6c: c5 7d 10 82 60 ff ff ff     	vmovupd	ymm8, ymmword ptr [rdx - 0xa0]
1400d3f74: c4 62 fd a8 fc              	vfmadd213pd	ymm15, ymm0, ymm4 # ymm15 = (ymm0 * ymm15) + ymm4
1400d3f79: c4 a2 7d 5a 44 11 26        	vbroadcasti128	ymm0, xmmword ptr [rcx + r10 + 0x26] # ymm0 = mem[0,1,0,1]
1400d3f80: c4 c2 7d 00 c5              	vpshufb	ymm0, ymm0, ymm13
1400d3f85: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
1400d3f89: c4 e2 7d 36 e5              	vpermd	ymm4, ymm0, ymm5
1400d3f8e: c5 dd fe c0                 	vpaddd	ymm0, ymm4, ymm0
1400d3f92: c5 fd 10 62 80              	vmovupd	ymm4, ymmword ptr [rdx - 0x80]
1400d3f97: c4 e2 a5 a8 c3              	vfmadd213pd	ymm0, ymm11, ymm3 # ymm0 = (ymm11 * ymm0) + ymm3
1400d3f9c: c4 a2 7d 5a 5c 11 32        	vbroadcasti128	ymm3, xmmword ptr [rcx + r10 + 0x32] # ymm3 = mem[0,1,0,1]
1400d3fa3: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400d3fa8: c4 41 30 57 c9              	vxorps	xmm9, xmm9, xmm9
1400d3fad: c4 62 65 36 ce              	vpermd	ymm9, ymm3, ymm6
1400d3fb2: c5 35 fe db                 	vpaddd	ymm11, ymm9, ymm3
1400d3fb6: c4 62 bd a8 df              	vfmadd213pd	ymm11, ymm8, ymm7 # ymm11 = (ymm8 * ymm11) + ymm7
1400d3fbb: c4 a2 7d 5a 5c 11 3e        	vbroadcasti128	ymm3, xmmword ptr [rcx + r10 + 0x3e] # ymm3 = mem[0,1,0,1]
1400d3fc2: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400d3fc7: c5 c0 57 ff                 	vxorps	xmm7, xmm7, xmm7
1400d3fcb: c4 e2 65 36 fe              	vpermd	ymm7, ymm3, ymm6
1400d3fd0: c5 c5 fe db                 	vpaddd	ymm3, ymm7, ymm3
1400d3fd4: c4 c2 dd a8 da              	vfmadd213pd	ymm3, ymm4, ymm10 # ymm3 = (ymm4 * ymm3) + ymm10
1400d3fd9: c5 fd 29 5c 24 20           	vmovapd	ymmword ptr [rsp + 0x20], ymm3
1400d3fdf: c4 a2 7d 5a 5c 11 4a        	vbroadcasti128	ymm3, xmmword ptr [rcx + r10 + 0x4a] # ymm3 = mem[0,1,0,1]
1400d3fe6: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400d3feb: c5 c0 57 ff                 	vxorps	xmm7, xmm7, xmm7
1400d3fef: c4 e2 65 36 fd              	vpermd	ymm7, ymm3, ymm5
1400d3ff4: c5 45 fe d3                 	vpaddd	ymm10, ymm7, ymm3
1400d3ff8: c4 62 bd a8 d2              	vfmadd213pd	ymm10, ymm8, ymm2 # ymm10 = (ymm8 * ymm10) + ymm2
1400d3ffd: c4 a2 7d 5a 54 11 56        	vbroadcasti128	ymm2, xmmword ptr [rcx + r10 + 0x56] # ymm2 = mem[0,1,0,1]
1400d4004: c4 c2 6d 00 d5              	vpshufb	ymm2, ymm2, ymm13
1400d4009: c5 e0 57 db                 	vxorps	xmm3, xmm3, xmm3
1400d400d: c4 e2 6d 36 dd              	vpermd	ymm3, ymm2, ymm5
1400d4012: c5 65 fe e2                 	vpaddd	ymm12, ymm3, ymm2
1400d4016: c4 62 dd a8 e1              	vfmadd213pd	ymm12, ymm4, ymm1 # ymm12 = (ymm4 * ymm12) + ymm1
1400d401b: c5 fd 10 4a a0              	vmovupd	ymm1, ymmword ptr [rdx - 0x60]
1400d4020: c5 fd 10 52 c0              	vmovupd	ymm2, ymmword ptr [rdx - 0x40]
1400d4025: c4 a2 7d 5a 5c 11 62        	vbroadcasti128	ymm3, xmmword ptr [rcx + r10 + 0x62] # ymm3 = mem[0,1,0,1]
1400d402c: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400d4031: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
1400d4035: c4 e2 65 36 e6              	vpermd	ymm4, ymm3, ymm6
1400d403a: c5 5d fe cb                 	vpaddd	ymm9, ymm4, ymm3
1400d403e: c4 62 f5 a8 0c 24           	vfmadd213pd	ymm9, ymm1, ymmword ptr [rsp] # ymm9 = (ymm1 * ymm9) + mem
1400d4044: c4 a2 7d 5a 5c 11 6e        	vbroadcasti128	ymm3, xmmword ptr [rcx + r10 + 0x6e] # ymm3 = mem[0,1,0,1]
1400d404b: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400d4050: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
1400d4054: c4 e2 65 36 e6              	vpermd	ymm4, ymm3, ymm6
1400d4059: c5 5d fe c3                 	vpaddd	ymm8, ymm4, ymm3
1400d405d: c4 42 ed a8 c6              	vfmadd213pd	ymm8, ymm2, ymm14 # ymm8 = (ymm2 * ymm8) + ymm14
1400d4062: c4 a2 7d 5a 5c 11 7a        	vbroadcasti128	ymm3, xmmword ptr [rcx + r10 + 0x7a] # ymm3 = mem[0,1,0,1]
1400d4069: c4 c2 65 00 dd              	vpshufb	ymm3, ymm3, ymm13
1400d406e: c5 d8 57 e4                 	vxorps	xmm4, xmm4, xmm4
1400d4072: c4 e2 65 36 e5              	vpermd	ymm4, ymm3, ymm5
1400d4077: c5 dd fe e3                 	vpaddd	ymm4, ymm4, ymm3
1400d407b: c4 c2 f5 a8 e7              	vfmadd213pd	ymm4, ymm1, ymm15 # ymm4 = (ymm1 * ymm4) + ymm15
1400d4080: c4 a2 7d 5a 8c 11 86 00 00 00       	vbroadcasti128	ymm1, xmmword ptr [rcx + r10 + 0x86] # ymm1 = mem[0,1,0,1]
1400d408a: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
1400d408f: c5 e0 57 db                 	vxorps	xmm3, xmm3, xmm3
1400d4093: c4 e2 75 36 dd              	vpermd	ymm3, ymm1, ymm5
1400d4098: c5 e5 fe d9                 	vpaddd	ymm3, ymm3, ymm1
1400d409c: c4 e2 ed a8 d8              	vfmadd213pd	ymm3, ymm2, ymm0 # ymm3 = (ymm2 * ymm3) + ymm0
1400d40a1: c5 fd 10 42 e0              	vmovupd	ymm0, ymmword ptr [rdx - 0x20]
1400d40a6: c4 a2 7d 5a 8c 11 92 00 00 00       	vbroadcasti128	ymm1, xmmword ptr [rcx + r10 + 0x92] # ymm1 = mem[0,1,0,1]
1400d40b0: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
1400d40b5: c5 e8 57 d2                 	vxorps	xmm2, xmm2, xmm2
1400d40b9: c4 e2 75 36 d6              	vpermd	ymm2, ymm1, ymm6
1400d40be: c5 ed fe f9                 	vpaddd	ymm7, ymm2, ymm1
1400d40c2: c4 c2 fd a8 fb              	vfmadd213pd	ymm7, ymm0, ymm11 # ymm7 = (ymm0 * ymm7) + ymm11
1400d40c7: c4 a2 7d 5a 8c 11 aa 00 00 00       	vbroadcasti128	ymm1, xmmword ptr [rcx + r10 + 0xaa] # ymm1 = mem[0,1,0,1]
1400d40d1: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
1400d40d6: c5 e8 57 d2                 	vxorps	xmm2, xmm2, xmm2
1400d40da: c4 e2 75 36 d5              	vpermd	ymm2, ymm1, ymm5
1400d40df: c5 ed fe d1                 	vpaddd	ymm2, ymm2, ymm1
1400d40e3: c4 c2 fd a8 d2              	vfmadd213pd	ymm2, ymm0, ymm10 # ymm2 = (ymm0 * ymm2) + ymm10
1400d40e8: c4 a2 7d 5a 84 11 9e 00 00 00       	vbroadcasti128	ymm0, xmmword ptr [rcx + r10 + 0x9e] # ymm0 = mem[0,1,0,1]
1400d40f2: c4 c2 7d 00 c5              	vpshufb	ymm0, ymm0, ymm13
1400d40f7: c5 f0 57 c9                 	vxorps	xmm1, xmm1, xmm1
1400d40fb: c4 e2 7d 36 ce              	vpermd	ymm1, ymm0, ymm6
1400d4100: c5 75 fe d0                 	vpaddd	ymm10, ymm1, ymm0
1400d4104: c5 fd 10 02                 	vmovupd	ymm0, ymmword ptr [rdx]
1400d4108: c4 62 fd a8 54 24 20        	vfmadd213pd	ymm10, ymm0, ymmword ptr [rsp + 0x20] # ymm10 = (ymm0 * ymm10) + mem
1400d410f: c4 a2 7d 5a 8c 11 b6 00 00 00       	vbroadcasti128	ymm1, xmmword ptr [rcx + r10 + 0xb6] # ymm1 = mem[0,1,0,1]
1400d4119: c4 c2 75 00 cd              	vpshufb	ymm1, ymm1, ymm13
1400d411e: c4 e2 75 36 ed              	vpermd	ymm5, ymm1, ymm5
1400d4123: c5 d5 fe c9                 	vpaddd	ymm1, ymm5, ymm1
1400d4127: c4 c2 fd a8 cc              	vfmadd213pd	ymm1, ymm0, ymm12 # ymm1 = (ymm0 * ymm1) + ymm12
1400d412c: 49 81 c2 c2 00 00 00        	add	r10, 0xc2
1400d4133: 48 81 c2 00 01 00 00        	add	rdx, 0x100
1400d413a: 4d 39 d0                    	cmp	r8, r10
1400d413d: 0f 85 9d fd ff ff           	jne	0x1400d3ee0 <.text+0xd2ee0>
