"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import type { UserInfo } from "@/lib/types";

interface InfoFormProps {
  onSubmit: (info: UserInfo) => void;
  onBack: () => void;
}

export default function InfoForm({ onSubmit, onBack }: InfoFormProps) {
  const [gender, setGender] = useState<"male" | "female" | "">("");
  const [age, setAge] = useState("");
  const [errors, setErrors] = useState<{ gender?: string; age?: string }>({});

  const validate = (): boolean => {
    const newErrors: { gender?: string; age?: string } = {};
    if (!gender) newErrors.gender = "성별을 선택해주세요";
    const ageNum = parseInt(age, 10);
    if (!age || isNaN(ageNum)) {
      newErrors.age = "나이를 입력해주세요";
    } else if (ageNum < 1 || ageNum > 100) {
      newErrors.age = "1~100세 사이의 나이를 입력해주세요";
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    onSubmit({
      gender: gender as "male" | "female",
      age: parseInt(age, 10),
    });
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-12">
      <div className="max-w-md w-full space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <p className="text-sm text-[var(--muted-foreground)]">STEP 1 / 3</p>
          <h2 className="text-2xl font-bold">기본 정보 입력</h2>
          <p className="text-sm text-[var(--muted-foreground)]">
            정확한 진단을 위해 기본 정보를 입력해주세요
          </p>
        </div>

        <Card className="shadow-sm border-0 bg-white">
          <CardHeader className="pb-4">
            <CardTitle className="text-lg">개인 정보</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Gender */}
              <div className="space-y-3">
                <Label className="text-sm font-medium">성별</Label>
                <RadioGroup
                  value={gender}
                  onValueChange={(v) => {
                    setGender(v as "male" | "female");
                    setErrors((prev) => ({ ...prev, gender: undefined }));
                  }}
                  className="grid grid-cols-2 gap-3"
                >
                  <Label
                    htmlFor="male"
                    className={`flex items-center justify-center gap-2 rounded-xl border-2 p-4 cursor-pointer transition-all ${
                      gender === "male"
                        ? "border-[var(--foreground)] bg-[var(--secondary)]"
                        : "border-[var(--border)] hover:border-[var(--muted-foreground)]"
                    }`}
                  >
                    <RadioGroupItem value="male" id="male" className="sr-only" />
                    <span className="text-xl">👨</span>
                    <span className="font-medium">남성</span>
                  </Label>
                  <Label
                    htmlFor="female"
                    className={`flex items-center justify-center gap-2 rounded-xl border-2 p-4 cursor-pointer transition-all ${
                      gender === "female"
                        ? "border-[var(--foreground)] bg-[var(--secondary)]"
                        : "border-[var(--border)] hover:border-[var(--muted-foreground)]"
                    }`}
                  >
                    <RadioGroupItem value="female" id="female" className="sr-only" />
                    <span className="text-xl">👩</span>
                    <span className="font-medium">여성</span>
                  </Label>
                </RadioGroup>
                {errors.gender && (
                  <p className="text-sm text-[var(--destructive)]">{errors.gender}</p>
                )}
              </div>

              {/* Age */}
              <div className="space-y-3">
                <Label htmlFor="age" className="text-sm font-medium">
                  나이
                </Label>
                <Input
                  id="age"
                  type="number"
                  inputMode="numeric"
                  placeholder="나이를 입력하세요"
                  min={1}
                  max={100}
                  value={age}
                  onChange={(e) => {
                    setAge(e.target.value);
                    setErrors((prev) => ({ ...prev, age: undefined }));
                  }}
                  className="h-12 rounded-xl text-center text-lg"
                />
                {errors.age && (
                  <p className="text-sm text-[var(--destructive)]">{errors.age}</p>
                )}
              </div>

              {/* Buttons */}
              <div className="flex gap-3 pt-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={onBack}
                  className="flex-1 h-12 rounded-xl"
                >
                  이전
                </Button>
                <Button
                  type="submit"
                  className="flex-[2] h-12 rounded-xl bg-[var(--foreground)] text-[var(--background)] hover:opacity-90"
                >
                  다음
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
