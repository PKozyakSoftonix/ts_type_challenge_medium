type MyReturnType<T> = T extends (...args: any[]) => infer R ? R : any;

type MyOmit<T, K extends keyof T> = {
  [P in keyof T as P extends K ? never : P]: T[P];
};

type MyReadonly2<T, K extends keyof T = keyof T> = {
  readonly [key in keyof T as key extends K ? key : never]: T[key];
} & {
  [key in keyof T as key extends K ? never : key]: T[key];
}

type DeepReadonly<T> = {
  readonly [K in keyof T]: keyof T[K] extends never ? T[K] : DeepReadonly<T[K]>
}

type TupleToUnion<T extends any[]> = T[number]

type Chainable<T = {}> = {
  option<K extends string, V extends any>(key: Exclude<K, keyof T>, value: V): Chainable<Omit<T,K> & Record<K, V>>;
  get(): T;
};

type Last<T extends any[]> = [any,...T][T["length"]]

type Pop<T extends any[]> = T extends [...infer I, infer P] ? I : []

//
type Awaited2<T> = T extends Promise<infer R> ? Awaited2<R> : T

declare function PromiseAll<T extends any[]>(values: readonly [...T]): Promise<{
  [P in keyof T]: Awaited2<T[P]>
}>
//

type LookUp<U, T extends string> = {
  [K in T]: U extends { type: T } ? U : never
}[T]

type TrimLeft<T extends string> = T extends `${' ' | '\t' | '\n'}${infer R}` ? TrimLeft<R> : T;

type TrimRight<T extends string> = T extends `${infer R}${' ' | '\t' | '\n'}` ? TrimRight<R> : T;

type Trim<T extends string> = T extends `${' ' | '\t' | '\n'}${infer R}` ? Trim<R> : TrimRight<T>;

type MyCapitalize<S extends string> = S extends `${infer x}${infer tail}` ? `${Uppercase<x>}${tail}` : S;

type Replace<S extends string, From extends string, To extends string> =
  From extends ''
    ? S
    : S extends `${infer V}${From}${infer R}`
      ? `${V}${To}${R}`
      : S

type ReplaceAll<S extends string, From extends string, To extends string> =
  From extends ''
    ? S
    : S extends `${infer V}${From}${infer R}`
      ?  `${V}${To}${ReplaceAll<`${R}`, From, To>}`
      : S

type AppendArgument<Fn, A> = Fn extends (...args: infer R) => infer T ? (...args: [...R, A]) => T : never

type Permutation<T, K=T> =
  [T] extends [never]
    ? []
    : K extends K
      ? [K, ...Permutation<Exclude<T, K>>]
      : never

//
type StrToArray<S extends string> = S extends `${infer First}${ infer Rest}` ? [1, ...StrToArray<Rest>]: []
type LengthOfString<S extends string> = StrToArray<S>['length'];
//

type Flatten<T extends any[]> = T extends [] ? [] : T extends [infer First, ...infer Rest]
  ? First extends any[] ? [...Flatten<First>, ...Flatten<Rest>] : [First, ...Flatten<Rest>]
  : never;

type AppendToObject<T, U extends keyof any, V> = {
  [K in keyof T | U]: K extends keyof T ? T[K] : V;
};

type Absolute<T extends number | string | bigint> = `${T}` extends `-${infer U}` ? U : `${T}`

type StringToUnion<T extends string> =  T extends `${infer First}${infer Rest}` ? First | StringToUnion<Rest> : never

type Merge<F, S> = {
  [K in keyof F | keyof S]: K extends keyof S
    ? S[K]
    : K extends keyof F
      ? F[K]
      : never;
};

type KebabCase<S extends string> = S extends `${infer S1}${infer S2}`
  ? S2 extends Uncapitalize<S2>
    ? `${Uncapitalize<S1>}${KebabCase<S2>}`
    : `${Uncapitalize<S1>}-${KebabCase<S2>}`
  : S;


//
// type Diff<T, P> =  {
//   [K in keyof (T & P) as K extends keyof (T | P) ? never : K]: (T & P)[K];
// };

type Diff<T, P> =  Omit<(T & P), keyof(T | P)>
//

type AnyOf<T extends any[]> = T[number] extends 0 | undefined | null | '' | false | [] | Record<string, never>
  ? false : true;

type IsNever<T> = [T] extends [never] ? true : false


type IsUnion<T, U = T> =
  [T] extends [never]
    ? false
    : T extends T
      ? [U] extends [T]
        ? false
        : true
      : false

type ReplaceKeys<U, T, Y> = {
  [K in keyof U]: K extends T
    ? K extends keyof Y
      ? Y[K]
      : never
    : U[K]
}


type RemoveIndexSignature<T, P=PropertyKey> = {
  [K in keyof T as P extends K
    ? never
    : K extends P
      ? K
      : never]: T[K]
}

//
type CheckPrefix<T> = T extends '+' | '-' ? T : never;
type CheckSuffix<T> =  T extends `${infer P}%` ? [P, '%'] : [T, ''];
type PercentageParser<A extends string> = A extends `${CheckPrefix<infer L>}${infer R}` ? [L, ...CheckSuffix<R>] : ['', ...CheckSuffix<A>];
//

type DropChar<S, C extends string> = S extends `${infer L}${C}${infer R}` ? DropChar<`${L}${R}`, C> : S;


type MinusOne<T extends number, A extends any[]=[1], P extends any[]=[]> =
  `${T}` extends 0
    ? -1
    : `${T}` extends A['length'] ? P['length'] : MinusOne<T, [...A, 1], A>


type PickByType<T extends Record<string, any>, U> = {
  [P in keyof T as T[P] extends U ? P : never]: T[P]
}

type StartsWith<T extends string, U extends string> =
  T extends `${U}${string}`
    ? true
    : false

type EndsWith<T extends string, U extends string> =
  T extends `${string}${U}`
    ? true
    : false

//
type IntersectionObj<T> = {
  [P in keyof T]: T[P];
}

type PartialByKeys<T, K extends keyof T = keyof T> = IntersectionObj<{
  [P in Exclude<keyof T, K>]: T[P];
} & {
  [P in K]+?: T[P];
}>;
//

type RequiredByKeys<T, K extends keyof T = keyof T> = IntersectionObj<{
  [P in keyof T as P extends K ? never : P]: T[P];
} & {
  [P in K]-?: T[P];
}>;

type Mutable<T> = {
  -readonly [K in keyof T]: T[K]
}

type OmitByType<T extends Record<string, any>, U> = {
  [P in keyof T as T[P] extends U ? never : P]: T[P]
}

type ObjectEntries<T, U = Required<T>> = {
  [K in keyof U]: [K, U[K] extends never ? undefined : U[K]]
}[keyof U]

type Shift<T extends any[]> = T extends [infer _, ...infer Rest] ? Rest : []

type TupleToNestedObject<T, U> = T extends [infer F,...infer R]
  ? { [K in F&string]:TupleToNestedObject<R,U> }
  :U

type Reverse<T extends any[], N extends any[] = []> =
  T extends [...infer Start, infer Last]
    ? [Last] extends []
      ? []
      : Reverse<Start, [...N, Last] >
    : N

type FlipArguments<T extends (...args : any)=> any> = T extends (...args: infer Args)=> infer U
  ? (...args: Reverse<Args>) => U
  : never;

type FlattenDepth<
  T extends any[],
  S extends number = 1,
  U extends any[] = []
  > = U['length'] extends S
  ? T
  : T extends [infer F, ...infer R]
    ? F extends any[]
      ? [...FlattenDepth<F, S, [...U, 1]>, ...FlattenDepth<R, S, U>]
      : [F, ...FlattenDepth<R, S, U>]
    : T

type BEM<B extends string, E extends string[],M extends string[]> = `${B}${E extends [] ? '' : `__${E[number]}`}${M extends [] ? '' : `--${M[number]}`}`

//
interface TreeNode {
  val: number
  left: TreeNode | null
  right: TreeNode | null
}
type InorderTraversal<T extends TreeNode | null> =
  [T] extends [TreeNode]
    ? [...InorderTraversal<T['left']>, T['val'], ...InorderTraversal<T['right']>]
    : [];
//

type Flip<T extends Record<string, any>> = {
  [key in keyof T as T[key] | `${T[key]}`]: key;
};

type Fibonacci<T extends number, CurrentIndex extends any[] = [1], Prev extends any[] = [], Current extends any[] = [1]> =
  CurrentIndex['length'] extends T
    ? Current['length']
    : Fibonacci<T, [...CurrentIndex, 1], Current, [...Prev, ...Current]>


type AllCombinations<
  S extends string,
  T extends string = StringToUnion<S>,
  U extends string = T,
  > = S extends `${infer _}${infer R}`
  ? U extends U
    ? `${U}${AllCombinations<R, U extends '' ? T : Exclude<T, U>>}`
    : never
  : ''

type GreaterThan<T extends number, U extends number, C extends any[] = []> =
  C['length'] extends T
    ? false
    : C['length'] extends U
      ? true
      : GreaterThan<T, U, [...C, 1]>

type Zip<T extends any[],U extends any[]> =
  T extends [infer TF,...infer TR]
    ? U extends [infer UF,...infer UR]
      ?[[TF,UF],...Zip<TR,UR>]
      :[]
    :[]

type IsTuple<T> =
  [T] extends [never]
    ? false
    : T extends ReadonlyArray<unknown>
      ? number extends T['length']
        ? false
        : true
      :false;

type Chunk<T extends any[], N extends number = 1, C extends any[] = []> =
  T extends [infer R, ...infer U]
    ? C['length'] extends N
      ? [C, ...Chunk<T, N>]
      : Chunk<U, N, [...C, R]>
    : C extends []
      ? C
      : [C]


type Fill<
  T extends unknown[],
  N,
  Start extends number = 0,
  End extends number = T['length'],
  L extends any[] = [],
  > = T extends [infer H, ...infer R]
  ? [...L, 0][Start] extends undefined
    ? Fill<R, N, Start, End, [...L, H]>
    : [...L, 0][End] extends undefined
      ? Fill<R, N, Start, End, [...L, N]>
      : Fill<R, N, Start, End, [...L, H]>
  : L

type ToUnion<U> = U extends any[] ? U[number] : U

type Without<T, U> =
  T extends [infer First, ...infer Rest]
    ? First extends ToUnion<U>
      ? Without<Rest, U>
      : [First, ...Without<Rest, U>]
    : T

type Trunc<T extends number | string> =
  `${T}` extends `${infer Main}.${infer _}`
    ? `${Main}` extends '' ? '0' : `${Main}`
    : `${T}`


type IsEqual<T, U> = U extends T ? T extends U ? true : false : false

type IndexOf<T extends any[], U, A extends any[] = []> =
  T extends [infer first, ...infer rest]
    ? IsEqual<first, U> extends true
      ? A['length']
      : IndexOf<rest, U, [...A, 0]>
    : -1


type Join<T extends any[], U extends string | number> =
  T extends [infer F, ...infer R]
    ? R['length'] extends 0
      ? `${F & string}`
      : `${F & string}${U}${Join<R, U>}`
    : ''


type LastIndexOf<T extends any[], U, A extends any[] = [], Res extends number = -1> =
  T extends [infer first, ...infer rest]
    ? IsEqual<first, U> extends true
      ? LastIndexOf<rest, U, [...A, 0], A['length']>
      : LastIndexOf<rest, U, [...A, 0], Res>
    : Res

type Includes<T, U> = U extends [infer F, ...infer Rest]
  ? IsEqual<F, T> extends true
    ? true
    : Includes<T, Rest>
  : false;

type Unique<T, U extends any[] = []> =
  T extends [infer R, ...infer F]
    ? Includes<R, U> extends true
      ? Unique<F, [...U]>
      : Unique<F, [...U, R]>
    : U

type MapTypes<T, R extends { mapFrom: any; mapTo: any }> = {
  [K in keyof T]: T[K] extends R['mapFrom']
    ? R extends { mapFrom: T[K] }
      ? R['mapTo']
      : never
    : T[K]
}

type ConstructTuple<L extends number, R extends unknown[] = []> = L extends R['length'] ? R : ConstructTuple<L, [...R, unknown]>


// type TupleToUnion<T extends any[]> = T[number]

type NumberRange<L, H, Count extends number[] = [], R extends number[] = []> =
  Count['length'] extends L
    ? NumberRange<L, H, [...Count, 0], [...R, Count['length']]>
    : Count['length'] extends H
      ? TupleToUnion<[...R, Count['length']]>
      : Count['length'] extends 0
        ? NumberRange<L, H, Count, R>
        : NumberRange<L, H, [...Count, 0], [...R, Count['length']]>


type Combination<T extends string[], All = T[number], Item = All> =
  Item extends string
    ? Item | `${Item} ${Combination<[], Exclude<All, Item>>}`
    : never

type Subsequence<T> = T extends [infer One, ...infer Rest]
  ? [One] | [...Subsequence<Rest>] | [One, ...Subsequence<Rest>]
  : []

type CheckRepeatedChars<T extends string> =
  T extends `${infer F}${infer E}`
    ? E extends `${string}${F}${string}`
      ? true
      : CheckRepeatedChars<E>
    : false


type FirstUniqueCharIndex<
  T extends string,
  U extends string[] = []
  > =
  T extends `${infer F}${infer R}`
    ? F extends U[number]
      ? FirstUniqueCharIndex<R, [...U, F]>
      : R extends `${string}${F}${string}`
        ? FirstUniqueCharIndex<R, [...U, F]>
        : U['length']
    : -1

type ParseUrlParams<T> =
  T extends `${string}:${infer R}`
    ? R extends `${infer P}/${infer L}`
      ? P | ParseUrlParams<L>
      : R
    : never


type GetMiddleElement<T extends any[]> =
  T['length'] extends 0 | 1 | 2
    ? T
    : T extends [any,...infer M,any]
      ? GetMiddleElement<M>
      : never


type FindEles<T extends unknown[], Appeared extends unknown[] = [], R extends unknown[] = []> =
  T extends [infer A, ...infer Rest]
    ? A extends [...Rest, ...Appeared][number]
      ? FindEles<Rest, [...Appeared, A], R>
      : FindEles<Rest, [...Appeared, A], [...R, A]>
    : R



type Flatten1<T,R extends any[] = []> =
  T extends [infer F,...infer L]
    ? [F] extends [never]
      ? Flatten1<L,R>
      : F extends any[]
        ? Flatten1<L,[...R,...Flatten1<F>]>
        : Flatten1<L,[...R,F]>
    :R


type Count1<T, R extends Record<string | number, any[]> = {}> =
  T extends [infer F, ...infer L]
    ? F extends keyof R
      ? Count1<L, Omit<R,F> & Record<F,[...R[F],0] > >
      : Count1<L, R & Record<F,[0]>>
    :{ [K in keyof R]:R[K]['length'] }


type CountElementNumberToObject<T> = Count1<Flatten1<T>>

type Integer<T extends number> = `${T}` extends `${bigint}` ? T : never

type ToPrimitive<T> =
  T extends Function
    ? Function
    : T extends object
      ? { [K in keyof T]: ToPrimitive<T[K]> }
      : T extends boolean
        ? boolean
        : T extends string
          ? string
          : T extends number
            ? number
            : never;


type DeepMutable<T extends Record<keyof any, any>> =
  T extends (...args:any[])=>any
    ? T
    : { - readonly [K in keyof T]: DeepMutable<T[K]> }

type All<T extends unknown[], K, Flag extends boolean = false> =
  T extends [infer Left, ...infer Rest]
    ? IsEqual<Left, K> extends true
      ? All<Rest, K, true>
      : false
    : Flag

type Filter<T extends unknown[], P> =
  T extends [infer F, ...infer R]
    ? F extends P
      ? [F, ...Filter<R, P>]
      : Filter<R, P>
    : [];

type FindAll<
  T extends string,
  S extends string,
  P extends any[] = [],
  R extends number[] = [],
  > = S extends '' ? []
  : P extends ''
    ? []
    : T extends `${string}${infer L}`
      ? T extends `${S}${string}`
        ? FindAll<L, S, [...P, 0], [...R, P['length']]>
        : FindAll<L, S, [...P, 0], R>
      :R

// type Combs<T extends string[]> =
//   T extends [infer F extends string, ...infer R extends string[]]
//   ? `${F} ${R[number]}` | Combs<R>
//   : never;


type PermutationsOfTuple<T extends unknown[], Prev extends unknown[] = []> =
  T extends [infer First, ...infer Rest]
    ? [First, ...PermutationsOfTuple<[...Prev, ...Rest]>] | (Rest extends [] ? never : PermutationsOfTuple<Rest, [...Prev, First]>)
    : []

type ReplaceFirst<T extends readonly unknown[], S, R> =
  T extends readonly [infer F, ...infer Rest]
    ? F extends S
      ? [R, ...Rest]
      : [F, ...ReplaceFirst<Rest, S, R>]
    : [];


type Transpose<M extends number[][], R = M['length'] extends 0 ? [] : M[0]> = {
  [X in keyof R]: {
    [Y in keyof M]: X extends keyof M[Y] ? M[Y][X] : never
  }
}

type Merge1<T> = {
  [K in keyof T]:T[K]
}

type RequireByKeys<T, KS extends keyof T> = Merge1< Required<Pick<T,KS>>& Omit<T,KS>>

type JSONSchema2TS<T> =
  T extends {type: "string"}
    ? T extends {enum:string[] }
      ? T['enum'][number]:string
    : T extends {type:"number"}
      ? T extends {enum:number[]}
        ? T['enum'][number]
        : number
      : T extends {type:"boolean"}
        ? boolean
        : T extends {type: "object"}
          ? T extends {properties:any}
            ? RequireByKeys<{[K in keyof T['properties']]?:JSONSchema2TS<T['properties'][K]>}, T extends {required:Array<keyof T['properties']>}?T['required'][number]: never >
            : Record<string,unknown>
          : T extends {type: "array"}
            ? T extends {items:any}
              ? Array<JSONSchema2TS<T['items']>>
              :unknown[]
            :never


type Multiplicate<N1 extends number, N2 extends number, C extends number[] = [], M extends number[] = [], R extends number[] = []> =
  C['length'] extends N1
    ? M['length'] extends N2
      ? R['length']
      : Multiplicate<N1, N2, C, [0, ...M], [...R, ...C]>
    : Multiplicate<N1, N2, [0, ...C], M, R>

type Square<N extends number, PN extends number = `${N}` extends `-${infer I extends number}` ? I : N> =
  PN extends 100
    ? 10000
    : Multiplicate<PN, PN>


type Triangular<N extends number, P extends number[] = [], A extends number[] = []> =
  P['length'] extends N
    ? A['length']
    : Triangular<N, [0, ...P], [...A, ...P, 0]>


type CartesianProduct<T, U> = T extends T
  ? U extends U
    ? [T, U]
    : never
  : never;

type MergeAll<XS extends object[], Res = {}> =
  XS extends [infer L, ...infer R extends object[]]
  ? MergeAll<R, Omit<Res, keyof L> & {
    [p in keyof L]: p extends keyof Res ? L[p] | Res[p] : L[p]
  }>
  : Omit<Res, never>;


type CheckRepeatedTuple<T extends unknown[]> =
  T extends [infer L, ...infer R]
    ? L extends R[number]
      ? true
      : CheckRepeatedTuple<R>
    : false

type PublicType<T extends Record<string, unknown>> = {
  [K in keyof T as K extends `_${string}` ? never : K]: T[K]
}

type DeepOmit<T, P extends string> =
  P extends `${infer K}.${infer Rest}`
    ? K extends keyof T
      ? { [Key in keyof T]: Key extends K
        ? DeepOmit<T[Key], Rest>
        : T[Key] }
      : T
    : Omit<T, P>

type IsOdd<T extends number> =  `${T}` extends `${number | ''}${1 | 3 | 5 | 7 | 9}` ? true : false;


type Hanoi<
  N extends number,
  From = 'A',
  To = 'B',
  Intermediate = 'C'> =
  Helper<N, [], From, To, Intermediate>

type Helper<
  N extends number,
  C extends 1[],
  From extends unknown,
  To extends unknown,
  Intermediate extends unknown>
  = C['length'] extends N
  ? []
  : [
    ...Helper<N, [...C, 1], From, Intermediate, To>,
    [From, To],
    ...Helper<N, [...C, 1], Intermediate, To, From>]

