; CF5 RVA=0xd3900 length=1341
; aliases: _ZN6micius8internal20GEMV_CF5PreparedRowsERKNS0_22PreparedCF5GEMVWeightsERKNS0_20PreparedF64GEMVInputEPfjj | micius::internal::GEMV_CF5PreparedRows
1400d3900: 55                          	push	rbp
1400d3901: 41 57                       	push	r15
1400d3903: 41 56                       	push	r14
1400d3905: 41 55                       	push	r13
1400d3907: 41 54                       	push	r12
1400d3909: 56                          	push	rsi
1400d390a: 57                          	push	rdi
1400d390b: 53                          	push	rbx
1400d390c: 48 81 ec d8 00 00 00        	sub	rsp, 0xd8
1400d3913: 48 8d ac 24 80 00 00 00     	lea	rbp, [rsp + 0x80]
1400d391b: c5 f9 29 7d 40              	vmovapd	xmmword ptr [rbp + 0x40], xmm7
1400d3920: c5 f9 7f 75 30              	vmovdqa	xmmword ptr [rbp + 0x30], xmm6
1400d3925: 4c 8b 29                    	mov	r13, qword ptr [rcx]
1400d3928: b0 01                       	mov	al, 0x1
1400d392a: 4d 85 ed                    	test	r13, r13
1400d392d: 0f 84 ec 04 00 00           	je	0x1400d3e1f <.text+0xd2e1f>
1400d3933: 48 8b 32                    	mov	rsi, qword ptr [rdx]
1400d3936: 48 85 f6                    	test	rsi, rsi
1400d3939: 0f 84 e0 04 00 00           	je	0x1400d3e1f <.text+0xd2e1f>
1400d393f: 8b 5a 08                    	mov	ebx, dword ptr [rdx + 0x8]
1400d3942: 48 85 db                    	test	rbx, rbx
1400d3945: 0f 84 d4 04 00 00           	je	0x1400d3e1f <.text+0xd2e1f>
1400d394b: 44 8b bd c0 00 00 00        	mov	r15d, dword ptr [rbp + 0xc0]
1400d3952: 45 39 cf                    	cmp	r15d, r9d
1400d3955: 0f 82 c4 04 00 00           	jb	0x1400d3e1f <.text+0xd2e1f>
1400d395b: 39 59 24                    	cmp	dword ptr [rcx + 0x24], ebx
1400d395e: 0f 85 bb 04 00 00           	jne	0x1400d3e1f <.text+0xd2e1f>
1400d3964: 44 8b 61 20                 	mov	r12d, dword ptr [rcx + 0x20]
1400d3968: 45 39 fc                    	cmp	r12d, r15d
1400d396b: 0f 82 ae 04 00 00           	jb	0x1400d3e1f <.text+0xd2e1f>
1400d3971: 44 8b 52 0c                 	mov	r10d, dword ptr [rdx + 0xc]
1400d3975: c5 f8 ae 5d e4              	vstmxcsr	dword ptr [rbp - 0x1c]
1400d397a: 41 bb c0 ff 00 00           	mov	r11d, 0xffc0
1400d3980: 44 23 5d e4                 	and	r11d, dword ptr [rbp - 0x1c]
1400d3984: 45 39 da                    	cmp	r10d, r11d
1400d3987: 0f 85 92 04 00 00           	jne	0x1400d3e1f <.text+0xd2e1f>
1400d398d: 44 89 ff                    	mov	edi, r15d
1400d3990: 44 29 cf                    	sub	edi, r9d
1400d3993: 75 07                       	jne	0x1400d399c <.text+0xd299c>
1400d3995: 31 c0                       	xor	eax, eax
1400d3997: e9 83 04 00 00              	jmp	0x1400d3e1f <.text+0xd2e1f>
1400d399c: 41 f6 c1 01                 	test	r9b, 0x1
1400d39a0: 0f 85 79 04 00 00           	jne	0x1400d3e1f <.text+0xd2e1f>
1400d39a6: 41 f6 c7 01                 	test	r15b, 0x1
1400d39aa: 74 09                       	je	0x1400d39b5 <.text+0xd29b5>
1400d39ac: 45 39 fc                    	cmp	r12d, r15d
1400d39af: 0f 85 6a 04 00 00           	jne	0x1400d3e1f <.text+0xd2e1f>
1400d39b5: 41 89 de                    	mov	r14d, ebx
1400d39b8: 41 c1 ee 05                 	shr	r14d, 0x5
1400d39bc: 41 89 da                    	mov	r10d, ebx
1400d39bf: 41 83 e2 1f                 	and	r10d, 0x1f
1400d39c3: 49 83 fa 01                 	cmp	r10, 0x1
1400d39c7: 4c 89 75 08                 	mov	qword ptr [rbp + 0x8], r14
1400d39cb: 49 83 de ff                 	sbb	r14, -0x1
1400d39cf: 4c 8b 59 08                 	mov	r11, qword ptr [rcx + 0x8]
1400d39d3: 4d 85 db                    	test	r11, r11
1400d39d6: 0f 84 43 04 00 00           	je	0x1400d3e1f <.text+0xd2e1f>
1400d39dc: 4d 89 e2                    	mov	r10, r12
1400d39df: 4c 0f af d3                 	imul	r10, rbx
1400d39e3: 4d 01 da                    	add	r10, r11
1400d39e6: 0f 82 33 04 00 00           	jb	0x1400d3e1f <.text+0xd2e1f>
1400d39ec: 4c 89 75 e8                 	mov	qword ptr [rbp - 0x18], r14
1400d39f0: 4c 8b 71 10                 	mov	r14, qword ptr [rcx + 0x10]
1400d39f4: 4d 85 f6                    	test	r14, r14
1400d39f7: 0f 84 22 04 00 00           	je	0x1400d3e1f <.text+0xd2e1f>
1400d39fd: 4c 89 5d f0                 	mov	qword ptr [rbp - 0x10], r11
1400d3a01: 4c 8b 5d e8                 	mov	r11, qword ptr [rbp - 0x18]
1400d3a05: 4d 0f af dc                 	imul	r11, r12
1400d3a09: 49 c1 e3 04                 	shl	r11, 0x4
1400d3a0d: 4c 89 75 f8                 	mov	qword ptr [rbp - 0x8], r14
1400d3a11: 4d 01 f3                    	add	r11, r14
1400d3a14: 0f 82 05 04 00 00           	jb	0x1400d3e1f <.text+0xd2e1f>
1400d3a1a: 48 8b 49 18                 	mov	rcx, qword ptr [rcx + 0x18]
1400d3a1e: 4c 01 e9                    	add	rcx, r13
1400d3a21: 0f 82 f8 03 00 00           	jb	0x1400d3e1f <.text+0xd2e1f>
1400d3a27: 4c 8d 34 dd 00 00 00 00     	lea	r14, [8*rbx]
1400d3a2f: 49 01 f6                    	add	r14, rsi
1400d3a32: 0f 82 e7 03 00 00           	jb	0x1400d3e1f <.text+0xd2e1f>
1400d3a38: 4c 89 4d 28                 	mov	qword ptr [rbp + 0x28], r9
1400d3a3c: 48 89 55 c0                 	mov	qword ptr [rbp - 0x40], rdx
1400d3a40: 4d 85 c0                    	test	r8, r8
1400d3a43: 0f 84 d6 03 00 00           	je	0x1400d3e1f <.text+0xd2e1f>
1400d3a49: 4a 8d 14 a5 00 00 00 00     	lea	rdx, [4*r12]
1400d3a51: 4c 89 45 20                 	mov	qword ptr [rbp + 0x20], r8
1400d3a55: 4c 01 c2                    	add	rdx, r8
1400d3a58: 0f 82 c1 03 00 00           	jb	0x1400d3e1f <.text+0xd2e1f>
1400d3a5e: 44 8b 4d 28                 	mov	r9d, dword ptr [rbp + 0x28]
1400d3a62: 4c 8b 45 20                 	mov	r8, qword ptr [rbp + 0x20]
1400d3a66: 4f 8d 04 88                 	lea	r8, [r8 + 4*r9]
1400d3a6a: 45 89 e1                    	mov	r9d, r12d
1400d3a6d: 45 29 f9                    	sub	r9d, r15d
1400d3a70: 49 c1 e1 02                 	shl	r9, 0x2
1400d3a74: 4c 29 ca                    	sub	rdx, r9
1400d3a77: 4d 39 d0                    	cmp	r8, r10
1400d3a7a: 73 0a                       	jae	0x1400d3a86 <.text+0xd2a86>
1400d3a7c: 48 3b 55 f0                 	cmp	rdx, qword ptr [rbp - 0x10]
1400d3a80: 0f 87 99 03 00 00           	ja	0x1400d3e1f <.text+0xd2e1f>
1400d3a86: 4d 39 d8                    	cmp	r8, r11
1400d3a89: 73 0a                       	jae	0x1400d3a95 <.text+0xd2a95>
1400d3a8b: 48 3b 55 f8                 	cmp	rdx, qword ptr [rbp - 0x8]
1400d3a8f: 0f 87 8a 03 00 00           	ja	0x1400d3e1f <.text+0xd2e1f>
1400d3a95: 49 39 c8                    	cmp	r8, rcx
1400d3a98: 73 09                       	jae	0x1400d3aa3 <.text+0xd2aa3>
1400d3a9a: 4c 39 ea                    	cmp	rdx, r13
1400d3a9d: 0f 87 7c 03 00 00           	ja	0x1400d3e1f <.text+0xd2e1f>
1400d3aa3: 4d 39 f0                    	cmp	r8, r14
1400d3aa6: 73 09                       	jae	0x1400d3ab1 <.text+0xd2ab1>
1400d3aa8: 48 39 f2                    	cmp	rdx, rsi
1400d3aab: 0f 87 6e 03 00 00           	ja	0x1400d3e1f <.text+0xd2e1f>
1400d3ab1: 83 ff 02                    	cmp	edi, 0x2
1400d3ab4: 0f 82 30 03 00 00           	jb	0x1400d3dea <.text+0xd2dea>
1400d3aba: 48 69 45 08 c2 00 00 00     	imul	rax, qword ptr [rbp + 0x8], 0xc2
1400d3ac2: 48 89 45 00                 	mov	qword ptr [rbp], rax
1400d3ac6: f6 c3 1f                    	test	bl, 0x1f
1400d3ac9: 0f 84 dd 00 00 00           	je	0x1400d3bac <.text+0xd2bac>
1400d3acf: 89 d8                       	mov	eax, ebx
1400d3ad1: 83 e0 e0                    	and	eax, -0x20
1400d3ad4: 48 89 45 d0                 	mov	qword ptr [rbp - 0x30], rax
1400d3ad8: 39 d8                       	cmp	eax, ebx
1400d3ada: 0f 85 98 01 00 00           	jne	0x1400d3c78 <.text+0xd2c78>
1400d3ae0: c4 e2 79 59 35 0f 66 01 00  	vpbroadcastq	xmm6, qword ptr [rip + 0x1660f] # 0x1400ea0f8
1400d3ae9: b3 3f                       	mov	bl, 0x3f
1400d3aeb: 49 be ff ff ff ff ff ff ef 7f       	movabs	r14, 0x7fefffffffffffff
1400d3af5: c5 fb 10 3d f3 65 01 00     	vmovsd	xmm7, qword ptr [rip + 0x165f3] # 0x1400ea0f0
1400d3afd: 48 8b 4d 28                 	mov	rcx, qword ptr [rbp + 0x28]
1400d3b01: d1 e9                       	shr	ecx
1400d3b03: 48 0f af 4d 00              	imul	rcx, qword ptr [rbp]
1400d3b08: 4c 01 e9                    	add	rcx, r13
1400d3b0b: 48 8d 45 18                 	lea	rax, [rbp + 0x18]
1400d3b0f: 48 89 44 24 20              	mov	qword ptr [rsp + 0x20], rax
1400d3b14: 48 89 f2                    	mov	rdx, rsi
1400d3b17: 4c 8b 45 08                 	mov	r8, qword ptr [rbp + 0x8]
1400d3b1b: 4c 8d 4d 10                 	lea	r9, [rbp + 0x10]
1400d3b1f: e8 1c 03 00 00              	call	0x1400d3e40 <.text+0xd2e40>
1400d3b24: c5 fa 7e 45 10              	vmovq	xmm0, qword ptr [rbp + 0x10]
1400d3b29: c4 e1 f9 7e c0              	vmovq	rax, xmm0
1400d3b2e: c4 e2 e0 f5 c0              	bzhi	rax, rax, rbx
1400d3b33: 4c 39 f0                    	cmp	rax, r14
1400d3b36: 0f 8f e1 02 00 00           	jg	0x1400d3e1d <.text+0xd2e1d>
1400d3b3c: c5 f9 db ce                 	vpand	xmm1, xmm0, xmm6
1400d3b40: c5 f9 2e f9                 	vucomisd	xmm7, xmm1
1400d3b44: 0f 82 d3 02 00 00           	jb	0x1400d3e1d <.text+0xd2e1d>
1400d3b4a: c5 fb 5a c0                 	vcvtsd2ss	xmm0, xmm0, xmm0
1400d3b4e: 8b 4d 28                    	mov	ecx, dword ptr [rbp + 0x28]
1400d3b51: 48 8b 45 20                 	mov	rax, qword ptr [rbp + 0x20]
1400d3b55: c5 fa 11 04 88              	vmovss	dword ptr [rax + 4*rcx], xmm0
1400d3b5a: c5 fa 7e 45 18              	vmovq	xmm0, qword ptr [rbp + 0x18]
1400d3b5f: c4 e1 f9 7e c0              	vmovq	rax, xmm0
1400d3b64: c4 e2 e0 f5 c0              	bzhi	rax, rax, rbx
1400d3b69: 4c 39 f0                    	cmp	rax, r14
1400d3b6c: 0f 8f ab 02 00 00           	jg	0x1400d3e1d <.text+0xd2e1d>
1400d3b72: c5 f9 db ce                 	vpand	xmm1, xmm0, xmm6
1400d3b76: c5 f9 2e f9                 	vucomisd	xmm7, xmm1
1400d3b7a: 0f 82 9d 02 00 00           	jb	0x1400d3e1d <.text+0xd2e1d>
1400d3b80: c5 fb 5a c0                 	vcvtsd2ss	xmm0, xmm0, xmm0
1400d3b84: 48 8b 4d 28                 	mov	rcx, qword ptr [rbp + 0x28]
1400d3b88: 8d 51 01                    	lea	edx, [rcx + 0x1]
1400d3b8b: 48 8b 45 20                 	mov	rax, qword ptr [rbp + 0x20]
1400d3b8f: c5 fa 11 04 90              	vmovss	dword ptr [rax + 4*rdx], xmm0
1400d3b94: 83 c1 02                    	add	ecx, 0x2
1400d3b97: 48 89 4d 28                 	mov	qword ptr [rbp + 0x28], rcx
1400d3b9b: 83 c7 fe                    	add	edi, -0x2
1400d3b9e: 83 ff 01                    	cmp	edi, 0x1
1400d3ba1: 0f 87 56 ff ff ff           	ja	0x1400d3afd <.text+0xd2afd>
1400d3ba7: e9 3e 02 00 00              	jmp	0x1400d3dea <.text+0xd2dea>
1400d3bac: c4 e2 79 59 35 43 65 01 00  	vpbroadcastq	xmm6, qword ptr [rip + 0x16543] # 0x1400ea0f8
1400d3bb5: b3 3f                       	mov	bl, 0x3f
1400d3bb7: 49 be ff ff ff ff ff ff ef 7f       	movabs	r14, 0x7fefffffffffffff
1400d3bc1: c5 fb 10 3d 27 65 01 00     	vmovsd	xmm7, qword ptr [rip + 0x16527] # 0x1400ea0f0
1400d3bc9: 48 8b 4d 28                 	mov	rcx, qword ptr [rbp + 0x28]
1400d3bcd: d1 e9                       	shr	ecx
1400d3bcf: 48 0f af 4d 00              	imul	rcx, qword ptr [rbp]
1400d3bd4: 4c 01 e9                    	add	rcx, r13
1400d3bd7: 48 8d 45 18                 	lea	rax, [rbp + 0x18]
1400d3bdb: 48 89 44 24 20              	mov	qword ptr [rsp + 0x20], rax
1400d3be0: 48 89 f2                    	mov	rdx, rsi
1400d3be3: 4c 8b 45 08                 	mov	r8, qword ptr [rbp + 0x8]
1400d3be7: 4c 8d 4d 10                 	lea	r9, [rbp + 0x10]
1400d3beb: e8 50 02 00 00              	call	0x1400d3e40 <.text+0xd2e40>
1400d3bf0: c5 fa 7e 45 10              	vmovq	xmm0, qword ptr [rbp + 0x10]
1400d3bf5: c4 e1 f9 7e c0              	vmovq	rax, xmm0
1400d3bfa: c4 e2 e0 f5 c0              	bzhi	rax, rax, rbx
1400d3bff: 4c 39 f0                    	cmp	rax, r14
1400d3c02: 0f 8f 15 02 00 00           	jg	0x1400d3e1d <.text+0xd2e1d>
1400d3c08: c5 f9 db ce                 	vpand	xmm1, xmm0, xmm6
1400d3c0c: c5 f9 2e f9                 	vucomisd	xmm7, xmm1
1400d3c10: 0f 82 07 02 00 00           	jb	0x1400d3e1d <.text+0xd2e1d>
1400d3c16: c5 fb 5a c0                 	vcvtsd2ss	xmm0, xmm0, xmm0
1400d3c1a: 8b 4d 28                    	mov	ecx, dword ptr [rbp + 0x28]
1400d3c1d: 48 8b 45 20                 	mov	rax, qword ptr [rbp + 0x20]
1400d3c21: c5 fa 11 04 88              	vmovss	dword ptr [rax + 4*rcx], xmm0
1400d3c26: c5 fa 7e 45 18              	vmovq	xmm0, qword ptr [rbp + 0x18]
1400d3c2b: c4 e1 f9 7e c0              	vmovq	rax, xmm0
1400d3c30: c4 e2 e0 f5 c0              	bzhi	rax, rax, rbx
1400d3c35: 4c 39 f0                    	cmp	rax, r14
1400d3c38: 0f 8f df 01 00 00           	jg	0x1400d3e1d <.text+0xd2e1d>
1400d3c3e: c5 f9 db ce                 	vpand	xmm1, xmm0, xmm6
1400d3c42: c5 f9 2e f9                 	vucomisd	xmm7, xmm1
1400d3c46: 0f 82 d1 01 00 00           	jb	0x1400d3e1d <.text+0xd2e1d>
1400d3c4c: c5 fb 5a c0                 	vcvtsd2ss	xmm0, xmm0, xmm0
1400d3c50: 48 8b 4d 28                 	mov	rcx, qword ptr [rbp + 0x28]
1400d3c54: 8d 51 01                    	lea	edx, [rcx + 0x1]
1400d3c57: 48 8b 45 20                 	mov	rax, qword ptr [rbp + 0x20]
1400d3c5b: c5 fa 11 04 90              	vmovss	dword ptr [rax + 4*rdx], xmm0
1400d3c60: 83 c1 02                    	add	ecx, 0x2
1400d3c63: 48 89 4d 28                 	mov	qword ptr [rbp + 0x28], rcx
1400d3c67: 83 c7 fe                    	add	edi, -0x2
1400d3c6a: 83 ff 01                    	cmp	edi, 0x1
1400d3c6d: 0f 87 56 ff ff ff           	ja	0x1400d3bc9 <.text+0xd2bc9>
1400d3c73: e9 72 01 00 00              	jmp	0x1400d3dea <.text+0xd2dea>
1400d3c78: 48 8b 45 08                 	mov	rax, qword ptr [rbp + 0x8]
1400d3c7c: c1 e0 04                    	shl	eax, 0x4
1400d3c7f: 48 03 45 f8                 	add	rax, qword ptr [rbp - 0x8]
1400d3c83: 48 89 45 d8                 	mov	qword ptr [rbp - 0x28], rax
1400d3c87: c5 fb 12 35 69 64 01 00     	vmovddup	xmm6, qword ptr [rip + 0x16469] # xmm6 = mem[0,0]
1400d3c8f: c5 fb 10 3d 59 64 01 00     	vmovsd	xmm7, qword ptr [rip + 0x16459] # 0x1400ea0f0
1400d3c97: 48 8b 4d 28                 	mov	rcx, qword ptr [rbp + 0x28]
1400d3c9b: 41 89 ce                    	mov	r14d, ecx
1400d3c9e: 49 8d 56 01                 	lea	rdx, [r14 + 0x1]
1400d3ca2: 48 0f af d3                 	imul	rdx, rbx
1400d3ca6: 48 8b 45 f0                 	mov	rax, qword ptr [rbp - 0x10]
1400d3caa: 48 01 c2                    	add	rdx, rax
1400d3cad: 48 89 55 c8                 	mov	qword ptr [rbp - 0x38], rdx
1400d3cb1: 48 89 df                    	mov	rdi, rbx
1400d3cb4: 49 0f af fe                 	imul	rdi, r14
1400d3cb8: 48 01 c7                    	add	rdi, rax
1400d3cbb: d1 e9                       	shr	ecx
1400d3cbd: 48 0f af 4d 00              	imul	rcx, qword ptr [rbp]
1400d3cc2: 4c 01 e9                    	add	rcx, r13
1400d3cc5: 48 8d 45 18                 	lea	rax, [rbp + 0x18]
1400d3cc9: 48 89 44 24 20              	mov	qword ptr [rsp + 0x20], rax
1400d3cce: 48 89 f2                    	mov	rdx, rsi
1400d3cd1: 4c 8b 45 08                 	mov	r8, qword ptr [rbp + 0x8]
1400d3cd5: 4c 8d 4d 10                 	lea	r9, [rbp + 0x10]
1400d3cd9: e8 62 01 00 00              	call	0x1400d3e40 <.text+0xd2e40>
1400d3cde: 48 8b 45 e8                 	mov	rax, qword ptr [rbp - 0x18]
1400d3ce2: 4c 89 f2                    	mov	rdx, r14
1400d3ce5: 49 0f af c6                 	imul	rax, r14
1400d3ce9: 48 c1 e0 04                 	shl	rax, 0x4
1400d3ced: 48 8b 4d d8                 	mov	rcx, qword ptr [rbp - 0x28]
1400d3cf1: c5 fa 10 04 01              	vmovss	xmm0, dword ptr [rcx + rax]
1400d3cf6: c5 fa 5a c8                 	vcvtss2sd	xmm1, xmm0, xmm0
1400d3cfa: c5 fb 10 45 10              	vmovsd	xmm0, qword ptr [rbp + 0x10]
1400d3cff: 48 8b 45 d0                 	mov	rax, qword ptr [rbp - 0x30]
1400d3d03: 0f be 0c 07                 	movsx	ecx, byte ptr [rdi + rax]
1400d3d07: c5 d3 2a d1                 	vcvtsi2sd	xmm2, xmm5, ecx
1400d3d0b: c5 eb 59 14 c6              	vmulsd	xmm2, xmm2, qword ptr [rsi + 8*rax]
1400d3d10: c5 eb 59 d1                 	vmulsd	xmm2, xmm2, xmm1
1400d3d14: c5 fb 58 c2                 	vaddsd	xmm0, xmm0, xmm2
1400d3d18: 48 ff c0                    	inc	rax
1400d3d1b: 48 39 d8                    	cmp	rax, rbx
1400d3d1e: 72 e3                       	jb	0x1400d3d03 <.text+0xd2d03>
1400d3d20: 48 89 d0                    	mov	rax, rdx
1400d3d23: 48 83 c8 01                 	or	rax, 0x1
1400d3d27: 48 0f af 45 e8              	imul	rax, qword ptr [rbp - 0x18]
1400d3d2c: 48 c1 e0 04                 	shl	rax, 0x4
1400d3d30: 48 8b 4d d8                 	mov	rcx, qword ptr [rbp - 0x28]
1400d3d34: c5 fa 10 0c 01              	vmovss	xmm1, dword ptr [rcx + rax]
1400d3d39: c5 f2 5a d1                 	vcvtss2sd	xmm2, xmm1, xmm1
1400d3d3d: c5 fb 10 4d 18              	vmovsd	xmm1, qword ptr [rbp + 0x18]
1400d3d42: 48 8b 45 d0                 	mov	rax, qword ptr [rbp - 0x30]
1400d3d46: 4c 8b 45 c8                 	mov	r8, qword ptr [rbp - 0x38]
1400d3d4a: 41 0f be 0c 00              	movsx	ecx, byte ptr [r8 + rax]
1400d3d4f: c5 d3 2a d9                 	vcvtsi2sd	xmm3, xmm5, ecx
1400d3d53: c5 e3 59 1c c6              	vmulsd	xmm3, xmm3, qword ptr [rsi + 8*rax]
1400d3d58: c5 e3 59 da                 	vmulsd	xmm3, xmm3, xmm2
1400d3d5c: c5 f3 58 cb                 	vaddsd	xmm1, xmm1, xmm3
1400d3d60: 48 ff c0                    	inc	rax
1400d3d63: 48 39 d8                    	cmp	rax, rbx
1400d3d66: 72 e2                       	jb	0x1400d3d4a <.text+0xd2d4a>
1400d3d68: c4 e1 f9 7e c0              	vmovq	rax, xmm0
1400d3d6d: b1 3f                       	mov	cl, 0x3f
1400d3d6f: c4 e2 f0 f5 c0              	bzhi	rax, rax, rcx
1400d3d74: 48 b9 ff ff ff ff ff ff ef 7f       	movabs	rcx, 0x7fefffffffffffff
1400d3d7e: 48 39 c8                    	cmp	rax, rcx
1400d3d81: 0f 8f 96 00 00 00           	jg	0x1400d3e1d <.text+0xd2e1d>
1400d3d87: c5 f9 54 d6                 	vandpd	xmm2, xmm0, xmm6
1400d3d8b: c5 f9 2e fa                 	vucomisd	xmm7, xmm2
1400d3d8f: 0f 82 88 00 00 00           	jb	0x1400d3e1d <.text+0xd2e1d>
1400d3d95: c5 fb 5a c0                 	vcvtsd2ss	xmm0, xmm0, xmm0
1400d3d99: 48 8b 45 20                 	mov	rax, qword ptr [rbp + 0x20]
1400d3d9d: c5 fa 11 04 90              	vmovss	dword ptr [rax + 4*rdx], xmm0
1400d3da2: c4 e1 f9 7e c8              	vmovq	rax, xmm1
1400d3da7: b1 3f                       	mov	cl, 0x3f
1400d3da9: c4 e2 f0 f5 c0              	bzhi	rax, rax, rcx
1400d3dae: 48 b9 ff ff ff ff ff ff ef 7f       	movabs	rcx, 0x7fefffffffffffff
1400d3db8: 48 39 c8                    	cmp	rax, rcx
1400d3dbb: 7f 60                       	jg	0x1400d3e1d <.text+0xd2e1d>
1400d3dbd: c5 f1 54 c6                 	vandpd	xmm0, xmm1, xmm6
1400d3dc1: c5 f9 2e f8                 	vucomisd	xmm7, xmm0
1400d3dc5: 72 56                       	jb	0x1400d3e1d <.text+0xd2e1d>
1400d3dc7: c5 f3 5a c1                 	vcvtsd2ss	xmm0, xmm1, xmm1
1400d3dcb: 48 8b 45 20                 	mov	rax, qword ptr [rbp + 0x20]
1400d3dcf: c5 fa 11 44 90 04           	vmovss	dword ptr [rax + 4*rdx + 0x4], xmm0
1400d3dd5: 8d 42 02                    	lea	eax, [rdx + 0x2]
1400d3dd8: 44 89 f9                    	mov	ecx, r15d
1400d3ddb: 48 89 45 28                 	mov	qword ptr [rbp + 0x28], rax
1400d3ddf: 29 c1                       	sub	ecx, eax
1400d3de1: 83 f9 01                    	cmp	ecx, 0x1
1400d3de4: 0f 87 ad fe ff ff           	ja	0x1400d3c97 <.text+0xd2c97>
1400d3dea: 44 39 7d 28                 	cmp	dword ptr [rbp + 0x28], r15d
1400d3dee: 0f 83 a1 fb ff ff           	jae	0x1400d3995 <.text+0xd2995>
1400d3df4: 44 89 7c 24 30              	mov	dword ptr [rsp + 0x30], r15d
1400d3df9: 48 8b 45 28                 	mov	rax, qword ptr [rbp + 0x28]
1400d3dfd: 89 44 24 28                 	mov	dword ptr [rsp + 0x28], eax
1400d3e01: 44 89 64 24 20              	mov	dword ptr [rsp + 0x20], r12d
1400d3e06: 48 8b 4d f0                 	mov	rcx, qword ptr [rbp - 0x10]
1400d3e0a: 48 8b 55 f8                 	mov	rdx, qword ptr [rbp - 0x8]
1400d3e0e: 4c 8b 45 c0                 	mov	r8, qword ptr [rbp - 0x40]
1400d3e12: 4c 8b 4d 20                 	mov	r9, qword ptr [rbp + 0x20]
1400d3e16: e8 45 e6 ff ff              	call	0x1400d2460 <.text+0xd1460>
1400d3e1b: eb 02                       	jmp	0x1400d3e1f <.text+0xd2e1f>
1400d3e1d: b0 03                       	mov	al, 0x3
1400d3e1f: c5 f8 28 75 30              	vmovaps	xmm6, xmmword ptr [rbp + 0x30]
1400d3e24: c5 f8 28 7d 40              	vmovaps	xmm7, xmmword ptr [rbp + 0x40]
1400d3e29: 48 81 c4 d8 00 00 00        	add	rsp, 0xd8
1400d3e30: 5b                          	pop	rbx
1400d3e31: 5f                          	pop	rdi
1400d3e32: 5e                          	pop	rsi
1400d3e33: 41 5c                       	pop	r12
1400d3e35: 41 5d                       	pop	r13
1400d3e37: 41 5e                       	pop	r14
1400d3e39: 41 5f                       	pop	r15
1400d3e3b: 5d                          	pop	rbp
1400d3e3c: c3                          	ret
