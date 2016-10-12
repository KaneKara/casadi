import subprocess
import re
import os

targets = set()
for L in file("src/Makefile","r"):
  m = re.match("ifeq \(\$\(TARGET\), ([A-Z0-9_]+)\)", L)
  if m:
    targets.add(m.group(1))

targets = ["X64_INTEL_HASWELL","X64_INTEL_SANDY_BRIDGE","X64_INTEL_CORE","X64_AMD_BULLDOZER","GENERIC"]

target_data = {}

this_pwd = os.getcwd()
for t in targets:
  p = subprocess.Popen(["make","CC=echo compiler_action `pwd`","TARGET=%s" % t],stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd="src")

  stdout,stderr = p.communicate()

  data = []

  for L in stdout.split("\n"):
    if L.startswith("compiler_action"):
      line = L.split(" ")
      pwd = line[1][len(this_pwd)+1:]
      cfile = line[-1]
      flags = [f for f in line[2:-4] if not f.startswith("-I")]
      data.append((cfile,pwd,flags))
      
  for cfile,pwd,f in data:
    if not( f==data[0][2] or len(f)==0):
      print "warning", f, data[0][2]
  target_data[t] = data

with file('CMakeLists.txt','w') as f:
  f.write("cmake_minimum_required(VERSION 2.8.6)\n")
  
  for t in targets:
    data = target_data[t]
    
    libname = "casadi_blasfeo_%s" % t
    
    f.write("casadi_external_library(%s %s)\n" % (libname, "\n".join([ os.path.join(pwd,cfile) for cfile, pwd, _ in data])))
    f.write("set_target_properties(%s PROPERTIES COMPILE_FLAGS \"%s\")\n\n" % (libname, " ".join(data[0][2])))
    for cfile, pwd, _ in data:
      if cfile.endswith("S"):
        f.write("set_property(SOURCE %s PROPERTY LANGUAGE C)\n" % os.path.join(pwd,cfile))

