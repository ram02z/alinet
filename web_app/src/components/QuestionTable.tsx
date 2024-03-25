import cx from 'clsx'
import { Table, Checkbox, ScrollArea, rem, Group, Center } from '@mantine/core'
import classes from './QuestionTable.module.css'
import { Question } from '../App'
import { useState } from 'react'
import {
  IconSelector,
  IconChevronDown,
  IconChevronUp,
} from '@tabler/icons-react'
export interface QuestionTableProps {
  selection: string[]
  setSelection: any
  questions: Question[]
  similarityThreshold: number
}

export const QuestionTable = ({
  selection,
  setSelection,
  questions,
  similarityThreshold,
}: QuestionTableProps) => {
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc' | null>(
    null
  )
  const Icon =
    sortDirection != null
      ? sortDirection === 'desc'
        ? IconChevronDown
        : IconChevronUp
      : IconSelector

  const sortedQuestions = [...questions].sort((a, b) => {
    if (sortDirection === 'asc') {
      return a.score - b.score
    }
    if (sortDirection === 'desc') {
      return b.score - a.score
    }
    return 0 // default order if not sorting
  })

  const toggleSortDirection = () => {
    setSortDirection((currentSortDirection) => {
      if (currentSortDirection === 'asc') {
        return 'desc'
      } else {
        return 'asc'
      }
    })
  }

  const toggleRow = (id: string) => {
    setSelection((current: string[]) =>
      current.includes(id)
        ? current.filter((item: string) => item !== id)
        : [...current, id]
    )
  }

  const toggleAll = () => {
    setSelection((current: string[]) =>
      current.length === questions.length
        ? []
        : questions.map((item) => item.id)
    )
  }

  const rows: any = sortedQuestions.map((item: Question) => {
    if (item.score >= similarityThreshold) {
      const selected = selection.includes(item.id)
      return (
        <Table.Tr
          key={item.id}
          className={cx({ [classes.rowSelected]: selected })}
        >
          <Table.Td>
            <Checkbox
              checked={selection.includes(item.id)}
              onChange={() => toggleRow(item.id)}
            />
          </Table.Td>
          <Table.Td>{item.text}</Table.Td>
          <Table.Td>{item.score}</Table.Td>
        </Table.Tr>
      )
    }
  })

  return (
    <ScrollArea>
      <Table verticalSpacing="md">
        <Table.Thead>
          <Table.Tr>
            <Table.Th style={{ width: rem(40) }}>
              <Checkbox
                onChange={toggleAll}
                checked={
                  questions.length > 0 && selection.length === questions.length
                }
                indeterminate={
                  selection.length > 0 && selection.length !== questions.length
                }
              />
            </Table.Th>
            <Table.Th>Questions</Table.Th>
            <Table.Th>
              <Group>
                Similarity Score
                <Center>
                  <Icon
                    onClick={toggleSortDirection}
                    style={{
                      width: rem(16),
                      height: rem(16),
                      cursor: 'pointer',
                    }}
                    stroke={1.5}
                  />
                </Center>
              </Group>
            </Table.Th>
          </Table.Tr>
        </Table.Thead>

        <Table.Tbody>{rows}</Table.Tbody>
      </Table>
    </ScrollArea>
  )
}
